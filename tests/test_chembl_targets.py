import networkx as nx
import pandas as pd

from src.ingestion.chembl_targets import (
    extract_gene_symbols,
    select_exact_molecules,
    select_exact_synonym_molecules,
)
from src.preprocessing.build_graph import _add_chembl_target_edges


def test_molecule_resolution_requires_exact_preferred_name():
    candidates = [
        {"molecule_chembl_id": "CHEMBL98", "pref_name": "VORINOSTAT"},
        {"molecule_chembl_id": "CHEMBL999", "pref_name": "VORINOSTAT ANALOG"},
        {"molecule_chembl_id": "CHEMBL998", "pref_name": "ANOTHER DRUG"},
    ]

    assert select_exact_molecules("vorinostat", candidates) == [candidates[0]]


def test_synonym_resolution_requires_exact_alias_match():
    candidates = [
        {
            "molecule_chembl_id": "CHEMBL3348846",
            "pref_name": "FEXAGRATINIB",
            "molecule_synonyms": [
                {"molecule_synonym": "AZD4547", "syn_type": "PND"},
                {"molecule_synonym": "AZD-4547", "syn_type": "RESEARCH_CODE"},
            ],
        }
    ]

    matches = select_exact_synonym_molecules("azd4547", candidates)

    assert len(matches) == 1
    assert matches[0][0]["molecule_chembl_id"] == "CHEMBL3348846"
    assert matches[0][1:] == ("AZD4547", "PND")
    assert select_exact_synonym_molecules("AZD454", candidates) == []


def test_target_symbols_come_from_gene_symbol_synonyms_only():
    target = {
        "target_components": [
            {
                "accession": "Q13547",
                "target_component_synonyms": [
                    {"component_synonym": "HDAC1", "syn_type": "GENE_SYMBOL"},
                    {"component_synonym": "RPD3L1", "syn_type": "GENE_SYMBOL_OTHER"},
                ],
            }
        ]
    }

    assert extract_gene_symbols(target) == [("Q13547", "HDAC1")]


def test_graph_preserves_exact_chembl_mechanism_provenance():
    evidence = pd.DataFrame(
        [
            {
                "gdsc_drug_id": "10",
                "gdsc_drug_name": "vorinostat",
                "molecule_chembl_id": "CHEMBL98",
                "target_chembl_id": "CHEMBL325",
                "target_name": "Histone deacetylase 1",
                "target_type": "SINGLE PROTEIN",
                "uniprot_accession": "Q13547",
                "gene_symbol": "HDAC1",
                "mechanism_of_action": "Histone deacetylase 1 inhibitor",
                "action_type": "INHIBITOR",
                "direct_interaction": True,
                "max_phase": 4,
                "source_url": "https://www.ebi.ac.uk/chembl/api/data/mechanism/121.json",
            }
        ]
    )
    graph = nx.Graph()

    _add_chembl_target_edges(graph, evidence)

    gdsc_drug = "GDSC_DRUG:10"
    molecule = "CHEMBL_MOLECULE:CHEMBL98"
    target = "CHEMBL_TARGET:CHEMBL325"
    assert graph[gdsc_drug][molecule]["relation"] == "exact_name_chembl_mapping"
    mechanism = graph[molecule][target]
    assert mechanism["relation"] == "chembl_direct_mechanism"
    assert mechanism["mechanism_of_action"] == "Histone deacetylase 1 inhibitor"
    assert graph[target]["HDAC1"]["relation"] == "has_chembl_gene_symbol"