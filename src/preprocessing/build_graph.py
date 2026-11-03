import os
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG


EDGE_ATTRIBUTE_NAMES = (
    "variant_count",
    "copy_number_count",
    "fusion_count",
    "ln_ic50",
    "auc",
    "copy_number_gain_count",
    "copy_number_loss_count",
    "copy_number_unclassified_count",
)


def _build_node_features(graph, node_mapping):
    node_types = sorted({attributes.get("type", "unknown") for _, attributes in graph.nodes(data=True)})
    if len(node_types) > 32:
        raise ValueError(f"Graph has {len(node_types)} node types; feature width is limited to 32")

    type_indices = {node_type: index for index, node_type in enumerate(node_types)}
    features = torch.zeros((len(node_mapping), 32), dtype=torch.float32)
    for node, index in node_mapping.items():
        node_type = graph.nodes[node].get("type", "unknown")
        features[index, type_indices[node_type]] = 1.0
    return features


def _add_mutation_edges(graph, mutations, clinical_samples):
    unmatched_samples = set(mutations["sample_id"]) - clinical_samples
    matched = mutations[
        mutations["sample_id"].isin(clinical_samples) & mutations["gene_symbol"].ne("")
    ]
    grouped = matched.groupby(["sample_id", "gene_symbol"], sort=False).size()
    for (sample, gene), variant_count in grouped.items():
        graph.add_node(sample, type="sample")
        graph.add_node(gene, type="gene")
        if graph.has_edge(sample, gene) and graph.edges[sample, gene].get("relation") == "has_gene_variant":
            graph.edges[sample, gene]["variant_count"] += int(variant_count)
        else:
            graph.add_edge(
                sample,
                gene,
                relation="has_gene_variant",
                variant_count=int(variant_count),
                copy_number_count=0,
                fusion_count=0,
            )
    return unmatched_samples


def _add_copy_number_summary(graph, copy_number_calls, sample_to_tumor):
    calls = copy_number_calls.copy()
    required_columns = {"sample_id", "gene_symbol", "status"}
    missing_columns = required_columns - set(calls.columns)
    if missing_columns:
        raise ValueError(
            f"Copy-number calls are missing required columns: {sorted(missing_columns)}"
        )
    calls["tumor_type"] = calls["sample_id"].map(sample_to_tumor).fillna("").replace("", "unclassified")
    calls["status"] = calls["status"].fillna("").astype(str).str.strip()
    calls = calls[calls["gene_symbol"].ne("")]
    grouped = calls.groupby(
        ["tumor_type", "gene_symbol", "status"], sort=False, dropna=False
    ).size()
    for (tumor, gene, status), call_count in grouped.items():
        normalized_status = status.casefold().replace("_", " ").replace("-", " ")
        if normalized_status in {"gain", "amplification"}:
            direction = "gain"
        elif normalized_status in {
            "loss",
            "deletion",
            "deep deletion",
            "deep loss",
        }:
            direction = "loss"
        else:
            direction = "unclassified"
        graph.add_node(tumor, type="tumor_type")
        graph.add_node(gene, type="gene")
        if not (
            graph.has_edge(tumor, gene)
            and graph.edges[tumor, gene].get("relation") == "copy_number_in_tumor"
        ):
            graph.add_edge(
                tumor,
                gene,
                relation="copy_number_in_tumor",
                variant_count=0,
                copy_number_count=0,
                fusion_count=0,
                copy_number_gain_count=0,
                copy_number_loss_count=0,
                copy_number_unclassified_count=0,
                copy_number_status_counts={},
            )
        attributes = graph.edges[tumor, gene]
        attributes["copy_number_count"] += int(call_count)
        direction_key = f"copy_number_{direction}_count"
        attributes[direction_key] += int(call_count)
        status_counts = attributes["copy_number_status_counts"]
        status_counts[status] = status_counts.get(status, 0) + int(call_count)


def _add_fusion_edges(graph, fusions, clinical_samples):
    unmatched_samples = set(fusions["sample_id"]) - clinical_samples
    matched = fusions[fusions["sample_id"].isin(clinical_samples)]
    for row in matched.itertuples(index=False):
        fusion_id = f"{row.sample_id}|{row.fusion_name}|{row.breakpoint_location}"
        graph.add_node(row.sample_id, type="sample")
        graph.add_node(fusion_id, type="fusion", fusion_name=row.fusion_name)
        graph.add_edge(
            row.sample_id,
            fusion_id,
            relation="has_fusion",
            variant_count=0,
            copy_number_count=0,
            fusion_count=1,
        )
        for gene in {row.gene_a, row.gene_b} - {"", "NA", "NaN"}:
            graph.add_node(gene, type="gene")
            graph.add_edge(
                fusion_id,
                gene,
                relation="fusion_gene",
                variant_count=0,
                copy_number_count=0,
                fusion_count=1,
            )
    return unmatched_samples


def _add_gdsc_response_edges(graph, responses):
    responses = responses.copy()
    responses["ln_ic50"] = pd.to_numeric(responses["ln_ic50"], errors="coerce")
    responses["auc"] = pd.to_numeric(responses["auc"], errors="coerce")
    responses = responses[
        responses["sanger_model_id"].ne("")
        & responses["drug_id"].ne("")
        & responses["ln_ic50"].notna()
        & responses["auc"].notna()
    ]
    group_columns = ["sanger_model_id", "cell_line", "reported_cancer_type", "drug_id", "drug_name"]
    aggregated = responses.groupby(group_columns, dropna=False, sort=False).agg(
        ln_ic50=("ln_ic50", "mean"),
        auc=("auc", "mean"),
        response_count=("ln_ic50", "size"),
        putative_target=("putative_target", "first"),
        pathway=("pathway", "first"),
    )
    for row in aggregated.itertuples():
        model_id, cell_line, reported_cancer_type, drug_id, drug_name = row.Index
        model_node = f"GDSC_MODEL:{model_id}"
        drug_node = f"GDSC_DRUG:{drug_id}"
        graph.add_node(
            model_node,
            type="cell_line",
            cell_line=cell_line,
            reported_cancer_type=reported_cancer_type,
        )
        graph.add_node(
            drug_node,
            type="drug",
            drug_name=drug_name,
            putative_target=row.putative_target,
            pathway=row.pathway,
        )
        if row.putative_target and str(row.putative_target).upper() != "NA":
            target_node = f"GDSC_TARGET_ANNOTATION:{row.putative_target}"
            graph.add_node(target_node, type="putative_target_annotation", label=row.putative_target)
            graph.add_edge(
                drug_node,
                target_node,
                relation="gdsc_putative_target_annotation",
                variant_count=0,
                copy_number_count=0,
                fusion_count=0,
            )
        if row.pathway and str(row.pathway).upper() != "NA":
            pathway_node = f"GDSC_PATHWAY_ANNOTATION:{row.pathway}"
            graph.add_node(pathway_node, type="putative_pathway_annotation", label=row.pathway)
            graph.add_edge(
                drug_node,
                pathway_node,
                relation="gdsc_putative_pathway_annotation",
                variant_count=0,
                copy_number_count=0,
                fusion_count=0,
            )
        graph.add_edge(
            drug_node,
            model_node,
            relation="gdsc2_response",
            variant_count=0,
            copy_number_count=0,
            fusion_count=0,
            ln_ic50=float(row.ln_ic50),
            auc=float(row.auc),
            response_count=int(row.response_count),
        )
        if reported_cancer_type:
            cancer_node = f"GDSC_CANCER:{reported_cancer_type}"
            graph.add_node(cancer_node, type="reported_cancer_type", label=reported_cancer_type)
            graph.add_edge(
                model_node,
                cancer_node,
                relation="has_reported_cancer_type",
                variant_count=0,
                copy_number_count=0,
                fusion_count=0,
            )

def _add_chembl_mapping_edges(graph, mappings):
    """Add every uniquely resolved drug-to-ChEMBL-molecule mapping.

    This relation is independent of whether a direct human-target mechanism
    exists. Mapping provenance is retained on the edge.
    """
    required = {
        "gdsc_drug_id", "gdsc_drug_name", "molecule_chembl_id",
        "drug_match_method", "matched_chembl_synonym",
        "chembl_synonym_type", "mapping_source_url", "provenance_state",
    }
    missing = required - set(mappings.columns)
    if missing:
        raise ValueError(f"ChEMBL molecule mapping table is missing columns: {sorted(missing)}")
    if mappings.empty:
        return
    if mappings["gdsc_drug_id"].astype(str).duplicated().any():
        raise ValueError("ChEMBL mapping table contains multiple molecules for one GDSC2 drug.")
    for row in mappings.itertuples(index=False):
        gdsc_drug = f"GDSC_DRUG:{row.gdsc_drug_id}"
        molecule = f"CHEMBL_MOLECULE:{row.molecule_chembl_id}"
        graph.add_node(
            gdsc_drug, type="drug", drug_name=row.gdsc_drug_name
        )
        graph.add_node(
            molecule, type="chembl_molecule",
            molecule_chembl_id=row.molecule_chembl_id,
        )
        graph.add_edge(
            gdsc_drug, molecule,
            relation="exact_name_chembl_mapping",
            drug_match_method=str(row.drug_match_method or "unknown"),
            matched_chembl_synonym=str(row.matched_chembl_synonym or ""),
            chembl_synonym_type=str(row.chembl_synonym_type or ""),
            mapping_source="ChEMBL molecule record",
            mapping_source_url=str(row.mapping_source_url),
            provenance_state=str(row.provenance_state),
            variant_count=0,
            copy_number_count=0,
            fusion_count=0,
        )


def _add_chembl_target_edges(graph, evidence):
    for row in evidence.itertuples(index=False):
        gdsc_drug = f"GDSC_DRUG:{row.gdsc_drug_id}"
        molecule = f"CHEMBL_MOLECULE:{row.molecule_chembl_id}"
        target = f"CHEMBL_TARGET:{row.target_chembl_id}"

        graph.add_node(
            gdsc_drug,
            type="drug",
            drug_name=row.gdsc_drug_name,
        )
        graph.add_node(
            molecule,
            type="chembl_molecule",
            molecule_chembl_id=row.molecule_chembl_id,
        )
        graph.add_node(
            target,
            type="target",
            target_name=row.target_name,
            target_type=row.target_type,
            uniprot_accession=row.uniprot_accession,
        )

        # Preserve how the GDSC drug was matched to ChEMBL.
        graph.add_edge(
            gdsc_drug,
            molecule,
            relation="exact_name_chembl_mapping",
            drug_match_method=str(
                getattr(row, "drug_match_method", "") or "unknown"
            ),
            matched_chembl_synonym=str(
                getattr(row, "matched_chembl_synonym", "") or ""
            ),
            chembl_synonym_type=str(
                getattr(row, "chembl_synonym_type", "") or ""
            ),
            mapping_source="ChEMBL molecule record",
            mapping_source_url=(
                f"https://www.ebi.ac.uk/chembl/api/data/"
                f"molecule/{row.molecule_chembl_id}.json"
            ),
            provenance_state="exact_string_match_not_chemical_equivalence",
            variant_count=0,
            copy_number_count=0,
            fusion_count=0,
        )

        # Preserve the mechanism record and its original source.
        graph.add_edge(
            molecule,
            target,
            relation="chembl_direct_mechanism",
            action_type=row.action_type,
            mechanism_of_action=row.mechanism_of_action,
            max_phase=row.max_phase,
            source_url=row.source_url,
            variant_count=0,
            copy_number_count=0,
            fusion_count=0,
        )

        if row.gene_symbol:
            graph.add_node(row.gene_symbol, type="gene")
            graph.add_edge(
                target,
                row.gene_symbol,
                relation="has_chembl_gene_symbol",
                variant_count=0,
                copy_number_count=0,
                fusion_count=0,
            )
def _edge_provenance_records(graph_edges):
    """Keep edge attributes aligned with the graph's edge ordering."""
    return [dict(attributes) for _, _, attributes in graph_edges]
def _edge_provenance_records(graph_edges):
    """Keep edge attributes aligned with the graph's edge ordering."""
    return [
        dict(attributes)
        for _, _, attributes in graph_edges
    ]


def _edge_attribute_values(attributes):
    return [
        float(attributes.get("variant_count", 0)),
        float(attributes.get("copy_number_count", 0)),
        float(attributes.get("fusion_count", 0)),
        float(attributes.get("ln_ic50", 0)),
        float(attributes.get("auc", 0)),
        float(attributes.get("copy_number_gain_count", 0)),
        float(attributes.get("copy_number_loss_count", 0)),
        float(attributes.get("copy_number_unclassified_count", 0)),
    ]
def build_openpedcan_graph():
    """Build a graph from OpenPedCan clinical metadata and consensus variants."""
    openpedcan_dir = Path(CONFIG["raw_data_path"]) / "openpedcan"
    raw_path = openpedcan_dir / "openpedcan_phase1.csv"
    mutation_path = openpedcan_dir / "openpedcan_mutations_phase2.csv"
    copy_number_path = openpedcan_dir / "openpedcan_copy_number_phase3.csv"
    fusion_path = openpedcan_dir / "openpedcan_fusions_phase4.csv"
    gdsc_path = Path(CONFIG["raw_data_path"]) / "gdsc" / "gdsc2_responses.csv"
    chembl_mapping_path = Path(CONFIG["raw_data_path"]) / "chembl" / "gdsc2_drug_molecule_mappings.csv"
    chembl_path = Path(CONFIG["raw_data_path"]) / "chembl" / "gdsc2_chembl_target_evidence.csv"
    if not all(path.is_file() for path in (raw_path, mutation_path, copy_number_path, fusion_path, gdsc_path, chembl_mapping_path, chembl_path)):
        raise FileNotFoundError("Build real OpenPedCan, GDSC2, ChEMBL molecule-mapping, and target-evidence tables before the graph")

    df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    G = nx.Graph()

    for _, row in df.iterrows():
        sample = str(row["sample_id"])
        tumor = str(row["tumor_type"])
        diagnosis = str(row["diagnosis"])

        G.add_node(sample, type="sample")
        if tumor:
            G.add_node(tumor, type="tumor_type")
            G.add_edge(sample, tumor, relation="classified_as")
        if diagnosis:
            G.add_node(diagnosis, type="diagnosis")
            G.add_edge(sample, diagnosis, relation="has_diagnosis")
        cohort = str(row.get("cohort", ""))
        if cohort:
            G.add_node(cohort, type="cohort")
            G.add_edge(sample, cohort, relation="from_cohort")
        subtype = str(row.get("molecular_subtype", ""))
        if subtype:
            G.add_node(subtype, type="molecular_subtype")
            G.add_edge(sample, subtype, relation="has_molecular_subtype")

    clinical_samples = set(df["sample_id"])
    mutation_columns = ["sample_id", "gene_symbol"]
    unmatched_samples: set[str] = set()
    for mutations in pd.read_csv(
        mutation_path,
        usecols=mutation_columns,
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):
        unmatched_samples.update(_add_mutation_edges(G, mutations, clinical_samples))

    if unmatched_samples:
        raise ValueError(f"Mutation table contains {len(unmatched_samples)} biospecimens absent from histologies.tsv")

    sample_to_tumor = dict(zip(df["sample_id"], df["tumor_type"]))
    copy_number_unmatched: set[str] = set()
    for calls in pd.read_csv(
        copy_number_path,
        usecols=["sample_id", "gene_symbol", "status"],
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):
        copy_number_unmatched.update(set(calls["sample_id"]) - clinical_samples)
        _add_copy_number_summary(G, calls, sample_to_tumor)
    if copy_number_unmatched:
        raise ValueError(f"Copy-number table contains {len(copy_number_unmatched)} biospecimens absent from histologies.tsv")

    fusions = pd.read_csv(
        fusion_path,
        usecols=["sample_id", "fusion_name", "gene_a", "gene_b", "breakpoint_location"],
        dtype=str,
        keep_default_na=False,
    )
    fusion_unmatched = _add_fusion_edges(G, fusions, clinical_samples)
    if fusion_unmatched:
        raise ValueError(f"Fusion table contains {len(fusion_unmatched)} biospecimens absent from histologies.tsv")

    gdsc_responses = pd.read_csv(gdsc_path, dtype=str, keep_default_na=False)
    _add_gdsc_response_edges(G, gdsc_responses)
    chembl_mappings = pd.read_csv(chembl_mapping_path, dtype=str, keep_default_na=False)
    _add_chembl_mapping_edges(G, chembl_mappings)
    chembl_evidence = pd.read_csv(chembl_path, dtype=str, keep_default_na=False)
    _add_chembl_target_edges(G, chembl_evidence)

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    graph_edges = list(G.edges(data=True))
    edge_index = torch.tensor(
        [[node_mapping[source], node_mapping[target]] for source, target, _ in graph_edges],
        dtype=torch.long,
    ).t().contiguous()
    edge_relations = sorted({attributes.get("relation", "unknown") for _, _, attributes in graph_edges})
    edge_relation_ids = {relation: index for index, relation in enumerate(edge_relations)}
    edge_type = torch.tensor(
        [edge_relation_ids[attributes.get("relation", "unknown")] for _, _, attributes in graph_edges],
        dtype=torch.long,
    )
    edge_weight = torch.tensor(
        [
            float(
                attributes.get("variant_count", 0)
                + attributes.get("copy_number_count", 0)
                + attributes.get("fusion_count", 0)
            ) or 1.0
            for _, _, attributes in graph_edges
        ],
        dtype=torch.float32,
    )
    edge_attr = torch.tensor(
        [_edge_attribute_values(attributes) for _, _, attributes in graph_edges],
        dtype=torch.float32,
    )
    x = _build_node_features(G, node_mapping)

    node_types = sorted({attributes.get("type", "unknown") for _, attributes in G.nodes(data=True)})
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        num_nodes=len(G.nodes()),
    )
    data.edge_provenance = _edge_provenance_records(graph_edges)
    data.node_ids = [str(node) for node in node_mapping]
    data.node_type_names = node_types
    data.edge_relation_names = edge_relations
    data.edge_attr_names = list(EDGE_ATTRIBUTE_NAMES)
    torch.save(data, str(processed_path / "openpedcan_graph.pt"))
    print(f"OpenPedCan graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_target_phase2_graph():
    """Build a minimal Phase 2 graph using TARGET molecular evidence."""
    raw_path = Path(CONFIG["raw_data_path"]) / "target" / "target_phase2.csv"
    if not raw_path.exists():
        print(f"⚠️  TARGET Phase 2 file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        sample = str(row["sample_id"])
        tumor = str(row["tumor_type"])
        gene = str(row["gene_symbol"])
        layer = str(row["molecular_layer"])

        G.add_node(sample, type="sample")
        G.add_node(tumor, type="tumor_type")
        G.add_node(gene, type="gene")
        G.add_node(layer, type="molecular_layer")

        G.add_edge(sample, tumor, relation="belongs_to")
        G.add_edge(tumor, gene, relation="has_target")
        G.add_edge(gene, layer, relation="has_layer")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "target_phase2_graph.pt"))
    print(f"✅ TARGET Phase 2 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_phase3_graph():
    """Build a minimal Phase 3 graph using GDSC2 + pediatric response evidence."""
    raw_path = Path(CONFIG["raw_data_path"]) / "gdsc_pediatric" / "gdsc_pediatric_phase3.csv"
    if not raw_path.exists():
        print(f"⚠️  GDSC2 + Pediatric Response Phase 3 file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        sample = str(row["sample_id"])
        tumor = str(row["tumor_type"])
        cell = str(row["cell_line"])
        drug = str(row["drug_name"])

        G.add_node(sample, type="sample")
        G.add_node(tumor, type="tumor_type")
        G.add_node(cell, type="cell_line")
        G.add_node(drug, type="drug")

        G.add_edge(sample, tumor, relation="belongs_to")
        G.add_edge(tumor, cell, relation="has_cell_line")
        G.add_edge(cell, drug, relation="response_to")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "phase3_graph.pt"))
    print(f"✅ Phase 3 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_phase4_graph():
    """Build a minimal Phase 4 graph using ChEMBL + Reactome + STRING evidence."""
    raw_path = Path(CONFIG["raw_data_path"]) / "mechanistic" / "chembl_reactome_string_phase4.csv"
    if not raw_path.exists():
        print(f"⚠️  Mechanistic Phase 4 file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        drug = str(row["drug_name"])
        gene = str(row["target_gene"])
        pathway = str(row["pathway"])
        interaction_type = str(row["interaction_type"])

        G.add_node(drug, type="drug")
        G.add_node(gene, type="gene")
        G.add_node(pathway, type="pathway")
        G.add_node(interaction_type, type="interaction")

        G.add_edge(drug, gene, relation="targets")
        G.add_edge(gene, pathway, relation="in_pathway")
        G.add_edge(drug, interaction_type, relation="interaction_type")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "phase4_graph.pt"))
    print(f"✅ Phase 4 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_phase5_graph():
    """Build a minimal Phase 5 graph using BindingDB + GO + Open Targets evidence."""
    raw_path = Path(CONFIG["raw_data_path"]) / "target_evidence" / "bindingdb_go_opentargets_phase5.csv"
    if not raw_path.exists():
        print(f"⚠️  Phase 5 target-evidence file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        drug = str(row["drug_name"])
        gene = str(row["target_gene"])
        go_term = str(row["gene_ontology_term"])
        evidence_type = str(row["evidence_type"])

        G.add_node(drug, type="drug")
        G.add_node(gene, type="gene")
        G.add_node(go_term, type="go_term")
        G.add_node(evidence_type, type="evidence")

        G.add_edge(drug, gene, relation="targets")
        G.add_edge(gene, go_term, relation="has_go_term")
        G.add_edge(drug, evidence_type, relation="evidence_type")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "phase5_graph.pt"))
    print(f"✅ Phase 5 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_phase6_graph():
    """Build a minimal Phase 6 graph using PDB structural-evidence data."""
    raw_path = Path(CONFIG["raw_data_path"]) / "structural" / "pdb_structural_phase6.csv"
    if not raw_path.exists():
        print(f"⚠️  Structural Phase 6 file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        drug = str(row["drug_name"])
        gene = str(row["target_gene"])
        pdb_id = str(row["pdb_id"])
        binding_mode = str(row["binding_mode"])

        G.add_node(drug, type="drug")
        G.add_node(gene, type="gene")
        G.add_node(pdb_id, type="structure")
        G.add_node(binding_mode, type="binding_mode")

        G.add_edge(drug, gene, relation="targets")
        G.add_edge(gene, pdb_id, relation="has_structure")
        G.add_edge(drug, binding_mode, relation="binds_in_mode")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "phase6_graph.pt"))
    print(f"✅ Phase 6 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_phase7_graph():
    """Build a minimal Phase 7 graph using experimental validation evidence."""
    raw_path = Path(CONFIG["raw_data_path"]) / "experimental" / "experimental_validation_phase7.csv"
    if not raw_path.exists():
        print(f"⚠️  Experimental Phase 7 file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        drug = str(row["drug_name"])
        tumor = str(row["tumor_type"])
        assay = str(row["assay_type"])
        result = str(row["validation_result"])

        G.add_node(drug, type="drug")
        G.add_node(tumor, type="tumor_type")
        G.add_node(assay, type="assay")
        G.add_node(result, type="validation_result")

        G.add_edge(drug, tumor, relation="tested_in")
        G.add_edge(drug, assay, relation="measured_by")
        G.add_edge(assay, result, relation="result")

    processed_path = Path(CONFIG["processed_data_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = _build_node_features(G, node_mapping)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, str(processed_path / "phase7_graph.pt"))
    print(f"✅ Phase 7 graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def build_heterogeneous_graph():
    raise RuntimeError("GDSC graph construction is disabled until verified real response data is configured")


if __name__ == "__main__":
    build_openpedcan_graph()