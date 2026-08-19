"""
Chemical representations: fingerprints, similarity and Butina clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina


def _parse_and_desalt(smiles: str) -> Chem.Mol | None:
    """
    Parse a SMILES string and reduce it to its largest organic fragment.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        return rdMolStandardize.LargestFragmentChooser().choose(molecule)
    except Exception:
        return None


def standardise_smiles(smiles_series: pd.Series) -> pd.DataFrame:
    """
    Standardise SMILES and derive an InChIKey for deduplication.

    Arguments
    ---------
    smiles_series: Raw SMILES indexed by compound name

    Returns
    -------
    DataFrame indexed by compound name with columns standard_smiles and
    inchikey
    """
    records = {}
    for compound_name, raw_smiles in smiles_series.items():
        molecule = _parse_and_desalt(raw_smiles) if isinstance(raw_smiles, str) else None
        if molecule is None:
            records[compound_name] = {"standard_smiles": np.nan, "inchikey": np.nan}
            continue
        records[compound_name] = {"standard_smiles": Chem.MolToSmiles(molecule),
                                  "inchikey": Chem.MolToInchiKey(molecule)}
    return pd.DataFrame.from_dict(records, orient="index")


def compute_ecfp4_fingerprints(smiles_series: pd.Series, 
                               n_bits: int = 2048, 
                               radius: int = 2) -> pd.DataFrame:
    """
    Compute ECFP4 fingerprints.

    Arguments
    ---------
    smiles_series: Standardised SMILES indexed by compound name.
    n_bits: Fingerprint length.

    Returns
    -------
    DataFrame of ECFP4 fingerprints, one row per compound, indexed by compound name.
    
    NOTE: compounds that fail to parse are dropped.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = {}
    for compound_name, smiles in smiles_series.items():
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            continue
        fingerprints[compound_name] = np.array(generator.GetFingerprint(molecule), dtype=np.uint8)

    return pd.DataFrame.from_dict(fingerprints, orient="index")


def compute_tanimoto_similarity_matrix(fingerprints: np.ndarray) -> np.ndarray:
    binary = fingerprints.astype(np.float32)
    intersection = binary @ binary.T
    bit_counts = binary.sum(axis=1)
    union = bit_counts[:, None] + bit_counts[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def cluster_by_tanimoto_butina(fingerprints: np.ndarray,
                               compound_names: list[str],
                               similarity_threshold: float = 0.4) -> pd.Series:
    """
    Group compounds into Butina clusters using Tanimoto distance.

    Arguments
    ---------
    fingerprints: (n_compounds, n_bits) binary array.
    compound_names: Names in the same order as the fingerprint rows.
    similarity_threshold: Tanimoto similarity cutoff for Butina clustering

    Returns
    -------
    pd.Series of integer cluster labels indexed by compound name.
    """
    similarity = compute_tanimoto_similarity_matrix(fingerprints)
    distance = 1.0 - similarity

    n_compounds = len(compound_names)
    condensed_distances = [float(distance[i, j]) for i in range(1, n_compounds) for j in range(i)]

    clusters = Butina.ClusterData(condensed_distances,
                                  n_compounds,
                                  1.0 - similarity_threshold,
                                  isDistData=True)

    labels = np.empty(n_compounds, dtype=int)
    for cluster_index, member_indices in enumerate(clusters):
        for member_index in member_indices:
            labels[member_index] = cluster_index

    return pd.Series(labels, index=pd.Index(compound_names), name="butina_cluster")