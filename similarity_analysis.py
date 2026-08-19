"""
Functions to assess the relationship between chemical and biological similarity.

Since comparing pairwise similarities leads to non-iid values, 
significance is assessed by permuting compound labels
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class SimilarityMatrices:
    """
    Dataclass for all-vs-all similarities in each of the three spaces
    """
    chemical: np.ndarray
    morphology: np.ndarray
    expression: np.ndarray
    compound_names: list[str]


def compute_cosine_similarity_matrix(profiles: np.ndarray) -> np.ndarray:
    """
    All-vs-all cosine similarity of row vectors.
    """
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    unit_vectors = profiles / np.clip(norms, 1e-12, None)
    return unit_vectors @ unit_vectors.T


def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    row_indices, column_indices = np.triu_indices_from(matrix, k=1)
    return matrix[row_indices, column_indices]


def compute_profile_magnitudes(profiles: np.ndarray) -> np.ndarray:
    """Return each compound's profile norm, used as a proxy for perturbation strength."""
    return np.linalg.norm(profiles, axis=1)


def _pairwise_mean_magnitude(magnitudes: np.ndarray) -> np.ndarray:
    """For each pair of compounds, get the mean of the two compounds' magnitudes."""
    pairwise_mean = (magnitudes[:, None] + magnitudes[None, :]) / 2.0
    return extract_upper_triangle(pairwise_mean)


def _residualise(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Remove the linear contribution of a covariate from a vector of values."""
    design = np.column_stack([np.ones_like(covariate), covariate])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def correlate_similarity_spaces(first_similarity: np.ndarray,
                                second_similarity: np.ndarray,
                                n_permutations: int = 1000,
                                random_seed: int = 0) -> dict[str, float]:
    """
    Correlate two similarity matrices, with a compound-level permutation test.

    The null distribution is built by shuffling compound labels of the second
    matrix, which preserves its internal structure while destroying any
    correspondence with the first. This respects the dependence between pairs
    that share a compound.

    Returns
    -------
    Dict with the observed Spearman correlation, the permutation p-value, and
    the mean and standard deviation of the null distribution.
    """
    observed = spearmanr(extract_upper_triangle(first_similarity),
                         extract_upper_triangle(second_similarity)).statistic

    generator = np.random.default_rng(random_seed)
    n_compounds = first_similarity.shape[0]
    first_pairs = extract_upper_triangle(first_similarity)

    null_correlations = np.empty(n_permutations)
    for permutation_index in range(n_permutations):
        shuffled = generator.permutation(n_compounds)
        permuted_matrix = second_similarity[np.ix_(shuffled, shuffled)]
        null_correlations[permutation_index] = spearmanr(
            first_pairs, extract_upper_triangle(permuted_matrix)).statistic

    n_at_least_as_extreme = int(np.sum(np.abs(null_correlations) >= abs(observed)))
    p_value = (n_at_least_as_extreme + 1) / (n_permutations + 1)

    return {"spearman": float(observed),
            "permutation_p": float(p_value),
            "null_mean": float(null_correlations.mean()),
            "null_sd": float(null_correlations.std())}


def summarise_biological_similarity_by_chemical_bin(chemical_similarity: np.ndarray,
                                                    biological_similarity: np.ndarray,
                                                    bin_edges: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)) -> pd.DataFrame:
    """
    Score mean biological similarity within bands of chemical similarity.

    The expected pattern is that highly similar molecules are reliably similar biologically,
    while below some Tanimoto threshold the relationship flattens out.
    """
    chemical_pairs = extract_upper_triangle(chemical_similarity)
    biological_pairs = extract_upper_triangle(biological_similarity)

    rows = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (chemical_pairs >= lower) & (chemical_pairs < upper)
        if not in_bin.any():
            continue
        rows.append(
            {"tanimoto_range": f"{lower:.1f}-{upper:.1f}",
             "n_pairs": int(in_bin.sum()),
             "mean_biological_similarity": float(biological_pairs[in_bin].mean()),
             "sd_biological_similarity": float(biological_pairs[in_bin].std())})
    return pd.DataFrame(rows)

def score_mechanism_retrieval(similarity: np.ndarray, mechanisms: pd.Series) -> dict[str, float]:
    """
    Measure how well a similarity space separates same-mechanism compound pairs.
    Reported as the AUC of using similarity to classify whether a pair shares a
    mechanism of action.

    Singleton mechanisms are excluded.
    """
    mechanism_counts = mechanisms.value_counts()
    keep = mechanisms.isin(mechanism_counts[mechanism_counts >= 2].index).to_numpy()

    filtered_similarity = similarity[np.ix_(keep, keep)]
    filtered_mechanisms = mechanisms[keep].to_numpy()

    same_mechanism = filtered_mechanisms[:, None] == filtered_mechanisms[None, :]

    similarity_pairs = extract_upper_triangle(filtered_similarity)
    label_pairs = extract_upper_triangle(same_mechanism.astype(float)) > 0

    positives = similarity_pairs[label_pairs]
    negatives = similarity_pairs[~label_pairs]

    ranks = pd.Series(similarity_pairs).rank().to_numpy()
    positive_rank_sum = ranks[label_pairs].sum()
    n_positive, n_negative = len(positives), len(negatives)
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)

    return {"auc": float(auc),
            "n_same_mechanism_pairs": int(n_positive),
            "n_different_mechanism_pairs": int(n_negative),
            "mean_similarity_same_mechanism": float(positives.mean()),
            "mean_similarity_different_mechanism": float(negatives.mean())}


def find_structurally_dissimilar_connections(matrices: SimilarityMatrices,
                                             mechanisms: pd.Series,
                                             cluster_labels: pd.Series,
                                             max_chemical_similarity: float = 0.3,
                                             min_biological_percentile: float = 99.0) -> pd.DataFrame:
    """
    Find chemically dissimilar pairs that are biologically similar in both assays.

    Chemical dissimilarity is required on two counts: a pairwise Tanimoto cutoff
    (the two molecules themselves are far apart) and membership of different Butina
    clusters (a cluster may be big enough to contain molecules that are quite
    dissimilar by themselves.)
    """
    row_indices, column_indices = np.triu_indices_from(matrices.chemical, k=1)

    chemical_pairs = matrices.chemical[row_indices, column_indices]
    morphology_pairs = matrices.morphology[row_indices, column_indices]
    expression_pairs = matrices.expression[row_indices, column_indices]

    morphology_threshold = np.percentile(morphology_pairs, min_biological_percentile)
    expression_threshold = np.percentile(expression_pairs, min_biological_percentile)

    labels = cluster_labels.loc[matrices.compound_names].to_numpy()
    in_different_clusters = labels[row_indices] != labels[column_indices]

    selected = ((chemical_pairs <= max_chemical_similarity)
                & in_different_clusters
                & (morphology_pairs >= morphology_threshold)
                & (expression_pairs >= expression_threshold))

    names = np.array(matrices.compound_names)
    hops = pd.DataFrame(
        {"compound_a": names[row_indices[selected]],
         "compound_b": names[column_indices[selected]],
         "tanimoto": chemical_pairs[selected],
         "morphology_similarity": morphology_pairs[selected],
         "expression_similarity": expression_pairs[selected]})

    hops["moa_a"] = mechanisms.loc[hops["compound_a"]].to_numpy()
    hops["moa_b"] = mechanisms.loc[hops["compound_b"]].to_numpy()
    hops["shares_mechanism"] = hops["moa_a"] == hops["moa_b"]

    return hops.sort_values("morphology_similarity", ascending=False).reset_index(drop=True)