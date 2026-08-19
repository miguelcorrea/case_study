"""
NOTE: Before running this script, run download_data.py

Script to test how much molecular similarity predicts biological response similarity.

The proposed model learns a compound representation shaped by two assays (LINCS L1000 and Cell Painting).
If chemical similarity already accounts for biological response similarity, a chemical fingerprint
is sufficient, and there is nothing for the learned representation to add.

This analysis uses the cpg0004-lincs dataset from Way et al. (2022) (https://doi.org/10.1016/j.cels.2022.10.001),
in which the same 1,571 compounds were profiled across 6 doses in A549 cells with both assays.

Grosso modo, the analysis carries out the following steps:
1. Match compounds across the two assays
2. Test agreement between assays
3. Test whether chemical similarity predicts biological similarity
4. Test how well each space separates known mechanisms of action
5. Look for chemically dissimilar compounds that are biologicaly similar

Usage
-----
python run_similarity_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chemistry import compute_ecfp4_fingerprints, compute_tanimoto_similarity_matrix, standardise_smiles, cluster_by_tanimoto_butina

from plotting import build_summary_figure

from data_loading import load_paired_dataset, standardise_features
from similarity_analysis import (SimilarityMatrices, compute_cosine_similarity_matrix, 
                                 correlate_similarity_spaces, find_structurally_dissimilar_connections,
                                 score_mechanism_retrieval,
                                 summarise_biological_similarity_by_chemical_bin)



def prepare_matched_compounds(data_directory, butina_cutoff=0.4) -> dict:
    """
    Join the two assays through shared chemical structures (one row per compound)
    """

    dataset = load_paired_dataset(data_directory)
    structures = standardise_smiles(dataset.annotations["smiles"])

    structures = structures[structures["standard_smiles"].notna()]
    n_before_dedup = len(structures)
    structures = structures[~structures["inchikey"].duplicated(keep="first")]
    n_duplicates = n_before_dedup - len(structures)
    print(f"Duplicate molecules removed: {n_duplicates}")

    fingerprints = compute_ecfp4_fingerprints(structures["standard_smiles"])
    fingerprint_array = fingerprints.to_numpy().astype(np.float32)
    compound_names = list(fingerprints.index)

    clusters = cluster_by_tanimoto_butina(fingerprint_array, compound_names, similarity_threshold=butina_cutoff)
    cluster_sizes = clusters.value_counts()

    dataset = dataset.subset(compound_names)

    morphology = standardise_features(dataset.morphology)
    expression = standardise_features(dataset.expression)

    return {
        "compound_names": compound_names,
        "fingerprints": fingerprint_array,
        "morphology": morphology.to_numpy().astype(np.float32),
        "expression": expression.to_numpy().astype(np.float32),
        "mechanisms": dataset.annotations["moa"],
        "clusters": clusters,
        "n_duplicates_removed": n_duplicates,
        "n_clusters": int(clusters.nunique()),
        "largest_cluster_size": int(cluster_sizes.iloc[0]),
        "n_singleton_clusters": int((cluster_sizes == 1).sum())}



def analyse_similarity_structure(matched: dict, n_permutations: int, random_seed: int) -> dict:
    """
    Carry out the similarity analyses.
    """

    matrices = SimilarityMatrices(chemical=compute_tanimoto_similarity_matrix(matched["fingerprints"]),
                                  morphology=compute_cosine_similarity_matrix(matched["morphology"]),
                                  expression=compute_cosine_similarity_matrix(matched["expression"]),
                                  compound_names=matched["compound_names"])

    #####################################################
    # Evaluate agreement between CP and gene expression #
    #####################################################

    print('Evaluating correlation between cell painting and gene expression data...')

    modality_agreement = correlate_similarity_spaces(matrices.morphology, matrices.expression, n_permutations, random_seed)

    print(f"Morphology vs expression: Spearman rho = {modality_agreement['spearman']:.3f}")
    print(f"Permutation null: {modality_agreement['null_mean']:.3f}")
    print(f"+/- {modality_agreement['null_sd']:.3f}, p = {modality_agreement['permutation_p']:.3f}")
        
    ########################################################################
    # Evaluate how well chemical similarity predicts biological similarity #
    ########################################################################

    print('Evaluating agreement between chemistry and biological similarity...')

    chemistry_vs_morphology = correlate_similarity_spaces(matrices.chemical, matrices.morphology, n_permutations, random_seed)
    chemistry_vs_expression = correlate_similarity_spaces(matrices.chemical, matrices.expression, n_permutations, random_seed)

    print(f"Tanimoto vs morphology Spearman rho: {chemistry_vs_morphology['spearman']:.3f}")
    print(f"Tanimoto vs expression Spearman rho: {chemistry_vs_expression['spearman']:.3f}")

    binned = summarise_biological_similarity_by_chemical_bin(matrices.chemical, matrices.morphology).merge(
             summarise_biological_similarity_by_chemical_bin(matrices.chemical, matrices.expression),
             on=["tanimoto_range", "n_pairs"],
             suffixes=("_morphology", "_expression"))
    print("Within bands of chemical similarity:")
    print(binned[["tanimoto_range","n_pairs",
                  "mean_biological_similarity_morphology",
                  "mean_biological_similarity_expression"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    #######################################################
    # Evaluate how well each space separates known MoAs   #
    #######################################################

    print('Evaluating how well each similarity space separates mechanisms of action...')

    mechanism_scores = {"chemical (Tanimoto)": score_mechanism_retrieval(matrices.chemical, matched["mechanisms"]),
                        "morphology (Cell Painting)": score_mechanism_retrieval(matrices.morphology, matched["mechanisms"]),
                        "expression (L1000)": score_mechanism_retrieval(matrices.expression, matched["mechanisms"])}

    print("AUC for separating same-mechanism from different-mechanism pairs:")
    for space, scores in mechanism_scores.items():
        print(f"{space}: {scores['auc']:.3f}")

    ############################################################################
    # Look for chemically dissimilar molecular with high biological similarity #
    ############################################################################

    structurally_dissimilar_connections = find_structurally_dissimilar_connections(matrices, matched["mechanisms"], matched["clusters"])
    n_sharing = int(structurally_dissimilar_connections["shares_mechanism"].sum())
    print(f"Pairs with Tanimoto similarity <0.3 but top 1% similarity in both assays: {len(structurally_dissimilar_connections)}\n"
          f"Of these, {n_sharing} share an annotated mechanism of action.")
    print("Highest-scoring examples:")
    print(structurally_dissimilar_connections.head(10)[["compound_a",
                         "compound_b",
                         "tanimoto","morphology_similarity",
                         "expression_similarity", "shares_mechanism"]].to_string(
                           index=False, float_format=lambda v: f"{v:.3f}"))

    return {"matrices": matrices,
            "modality_agreement": modality_agreement,
            "chemistry_vs_morphology": chemistry_vs_morphology,
            "chemistry_vs_expression": chemistry_vs_expression,
            "binned_similarity": binned,
            "mechanism_scores": mechanism_scores,
            "structurally_unrelated_connections": structurally_dissimilar_connections}


def save_outputs(matched: dict, analysis: dict, results_directory: Path) -> None:
    """Write output tables and a machine-readable summary."""

    analysis["binned_similarity"].to_csv(results_directory / "biological_similarity_by_tanimoto_bin.csv", index=False)
    analysis["structurally_unrelated_connections"].to_csv(results_directory / "structurally_unrelated_connections.csv", index=False)

    summary = {"n_compounds": len(matched["compound_names"]),
               "n_duplicate_molecules_removed": matched["n_duplicates_removed"],
               "modality_agreement": analysis["modality_agreement"],
               "chemistry_vs_morphology": analysis["chemistry_vs_morphology"],
               "chemistry_vs_expression": analysis["chemistry_vs_expression"],
               "mechanism_auc": {space: scores["auc"] for space, scores in analysis["mechanism_scores"].items()},
               "n_structurally_unrelated_connections": int(len(analysis["structurally_unrelated_connections"])),
               "n_structurally_unrelated_connections_sharing_mechanism": int(analysis["structurally_unrelated_connections"]["shares_mechanism"].sum())}

    with open(results_directory / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    figure_path = build_summary_figure(analysis["matrices"].chemical,
                                       analysis["matrices"].morphology,
                                       analysis["matrices"].expression,
                                       analysis["mechanism_scores"],
                                       results_directory / "similarity_analysis.png")
    


def main(data_directory, results_directory, n_permutations, random_seed) -> None:
    matched = prepare_matched_compounds(data_directory)
    analysis = analyse_similarity_structure(matched, n_permutations, random_seed)
    save_outputs(matched, analysis, results_directory)


if __name__ == "__main__":

    data_directory = Path('data')
    results_directory = Path('results')
    results_directory.mkdir(exist_ok=True)
    n_permutations = 500
    random_seed = 42

    main(data_directory, results_directory, n_permutations, random_seed)