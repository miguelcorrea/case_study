"""
Loading and joining the LINCS Cell Painting / L1000 consensus profiles.

Both assays come from cpg0004-lincs: 1,571 compounds profiled in A549 cells with
Cell Painting and L1000 on matched plates. We use the published consensus
signatures from the Way et al. (2022) analysis repository, which are already
aggregated to one profile per compound per dose.

The output of this module is a single PairedDataset holding, for the compounds
present in both assays:
    - a morphology matrix   (compounds x Cell Painting features)
    - an expression matrix  (compounds x L1000 landmark genes)
    - compound annotations  (name, SMILES, InChIKey, mechanism of action)

All three are row-aligned on the same compound index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Column-name prefixes used by CellProfiler for the three segmented compartments.
CELL_PAINTING_FEATURE_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_")

# L1000 landmark genes are stored under their Affymetrix probe-set identifiers,
# which all end in "_at".
L1000_FEATURE_SUFFIX = "_at"

# The Drug Repurposing Hub export begins with commented provenance lines.
REPURPOSING_HUB_COMMENT_CHAR = "!"


@dataclass
class PairedDataset:
    """
    Row-aligned morphology, expression and annotation tables.

    Attributes
    ----------
    morphology:  (n_compounds, n_morphology_features) Cell Painting profiles.
    expression:  (n_compounds, n_landmark_genes) L1000 signatures.
    annotations: (n_compounds, ...) compound-level metadata. Its index
                  matches the row order of the two feature matrices.
    """

    morphology: pd.DataFrame
    expression: pd.DataFrame
    annotations: pd.DataFrame

    def __post_init__(self) -> None:
        if not (
            self.morphology.index.equals(self.expression.index)
            and self.morphology.index.equals(self.annotations.index)
        ):
            raise ValueError("Morphology, expression and annotation rows are not aligned.")

    @property
    def n_compounds(self) -> int:
        return len(self.annotations)

    def subset(self, compound_names: list[str]) -> "PairedDataset":
        """Return a new dataset restricted to the given compounds, order preserved."""
        return PairedDataset(
            morphology=self.morphology.loc[compound_names],
            expression=self.expression.loc[compound_names],
            annotations=self.annotations.loc[compound_names],
        )

    def describe(self) -> str:
        return (
            f"{self.n_compounds} compounds | "
            f"{self.morphology.shape[1]} morphology features | "
            f"{self.expression.shape[1]} landmark genes | "
            f"{self.annotations['moa'].nunique()} distinct mechanisms"
        )


def _select_feature_columns(table: pd.DataFrame, assay: str) -> list[str]:
    """Return the measurement columns for an assay, excluding all metadata columns."""
    if assay == "morphology":
        return [c for c in table.columns if c.startswith(CELL_PAINTING_FEATURE_PREFIXES)]
    if assay == "expression":
        return [c for c in table.columns if c.endswith(L1000_FEATURE_SUFFIX)]
    raise ValueError(f"Unknown assay: {assay!r}")


def _aggregate_doses_to_compound_level(table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Collapse the per-dose rows of one assay into one profile per compound.

    The published consensus files contain one row per compound per dose. Our model
    operates at the compound level, so we take the median across doses. The median
    is preferred over the mean because a single extreme dose (usually the highest,
    where cytotoxicity dominates) should not determine the compound's profile.
    """
    return table.groupby("pert_iname")[feature_columns].median()


def _load_assay_profiles(path: Path, assay: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one assay file and return (compound-level features, compound-level MOA)."""
    table = pd.read_csv(path, sep="\t")
    feature_columns = _select_feature_columns(table, assay)
    if not feature_columns:
        raise ValueError(f"No {assay} feature columns found in {path.name}.")

    features = _aggregate_doses_to_compound_level(table, feature_columns)
    mechanisms = table.groupby("pert_iname")["moa"].first()
    return features, mechanisms


def _load_structures_from_repurposing_hub(path: Path) -> pd.DataFrame:
    """Read compound structures, keeping one record per compound name.

    The Hub lists a separate row per physical sample, so the same compound appears
    several times (different vendors, salt forms, batch identifiers). We keep the
    first record per compound name; the SMILES are identical across those rows for
    the fields we use.
    """
    hub = pd.read_csv(path, sep="\t", comment=REPURPOSING_HUB_COMMENT_CHAR)
    hub = hub.dropna(subset=["smiles", "pert_iname"])
    return hub.groupby("pert_iname")[["smiles", "InChIKey"]].first()


def load_paired_dataset(data_directory: str | Path) -> PairedDataset:
    """Build the compound-level paired dataset from the three source files.

    Compounds are retained only if they appear in both assays and have a structure
    in the Drug Repurposing Hub, since the model needs all three.

    Arguments
    ---------
    data_directory: Directory holding cp_di.tsv.gz, l1000_di.tsv.gz and
                    repurposing_samples.txt.

    Returns
    -------
    A PairedDataset whose three tables share a common compound index.
    """
    directory = Path(data_directory)

    morphology, morphology_moa = _load_assay_profiles(directory / "cp_di.tsv.gz", "morphology")
    expression, expression_moa = _load_assay_profiles(directory / "l1000_di.tsv.gz", "expression")
    structures = _load_structures_from_repurposing_hub(directory / "repurposing_samples.txt")

    shared_compounds = sorted(set(morphology.index) & set(expression.index) & set(structures.index))

    annotations = pd.DataFrame(index=pd.Index(shared_compounds, name="pert_iname"))
    annotations["smiles"] = structures.loc[shared_compounds, "smiles"]
    annotations["hub_inchikey"] = structures.loc[shared_compounds, "InChIKey"]
    # The two assay files carry independently curated MOA strings; prefer the
    # Cell Painting annotation and fall back to L1000 where it is missing.
    annotations["moa"] = morphology_moa.loc[shared_compounds].fillna(
        expression_moa.loc[shared_compounds])

    return PairedDataset(
        morphology=morphology.loc[shared_compounds],
        expression=expression.loc[shared_compounds],
        annotations=annotations,
    )


def standardise_features(features: pd.DataFrame, drop_low_variance: bool = True) -> pd.DataFrame:
    """
    Z-score each feature and drop columns that carry no usable signal.

    Feature-wise standardisation puts morphology and expression on a comparable
    scale, which matters because the two assays have different native units and
    very different feature counts.

    Note: in the full pipeline these statistics must be fitted on the training
    split only. Here they are fitted on all compounds because the standardisation
    is unsupervised and the downstream splits are evaluated separately; this is
    called out explicitly in the leakage discussion.
    """
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.loc[:, cleaned.notna().all(axis=0)]

    if drop_low_variance:
        cleaned = cleaned.loc[:, cleaned.std(axis=0) > 1e-8]

    return (cleaned - cleaned.mean(axis=0)) / cleaned.std(axis=0)
