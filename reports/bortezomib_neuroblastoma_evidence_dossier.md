# Bortezomib–neuroblastoma evidence dossier

**Prepared:** 2026-11-02
**Purpose:** Assess what the local knowledge-graph lead actually shows, check existing neuroblastoma prior art, and define a defensible public-data follow-up. This is a research prioritization document, not a treatment recommendation.

## Executive summary

The graph lead is reproducible but should be described as **bortezomib → one ChEMBL 26S proteasome complex record → 38 component-gene annotations that overlap Neuroblastoma-labelled OpenPedCan variant records**. It is not 38 independent drug-target validations, and gene-symbol overlap does not establish target dependency, functional alteration, or drug sensitivity.

Bortezomib has already been tested preclinically in neuroblastoma and in early-phase pediatric combination trials. Therefore, “test bortezomib in neuroblastoma” is not a novel research question. A possible, unvalidated angle is whether a pre-specified proteasome-complex molecular feature stratifies bortezomib response across neuroblastoma models. Neither the GDSC2 analysis nor the small exploratory CTRP analysis provides statistically persuasive evidence for the 38-gene expression score as a response biomarker.

**Conclusion:** continue only as a hypothesis-generating, public-data analysis. Do not claim novelty or efficacy from the graph overlap.

## Reproducible local evidence

### Drug-to-target record

The local research index records:

| Link | Record |
|---|---|
| GDSC2 drug | Drug ID `1191`, Bortezomib |
| ChEMBL molecule | `CHEMBL325041`, selected by `exact_preferred_name` |
| ChEMBL mechanism/target | `CHEMBL2364701`, “26S proteasome,” target type `PROTEIN COMPLEX` |
| Mechanism annotation | “26S proteasome inhibitor,” `INHIBITOR`, direct interaction `True`, max phase `4` |
| Component annotations | 38 distinct UniProt accessions and 38 gene symbols |

The ChEMBL evidence is from the local extract; its recorded mechanism URL is [ChEMBL mechanism record 1431](https://www.ebi.ac.uk/chembl/api/data/mechanism/1431.json). The exact preferred-name mapping is an identifier mapping, not a structure-based chemical-equivalence check.

The 38 component symbols are:

`ADRM1, PSMA1–PSMA8, PSMB1–PSMB11, PSMC1–PSMC6, PSMD1, PSMD2, PSMD3, PSMD4, PSMD6, PSMD7, PSMD8, PSMD11, PSMD12, PSMD13, PSMD14, SEM1`.

The compound maps to one target complex; showing each subunit as a “bortezomib target” would overstate what this record says.

### OpenPedCan overlap

The source is the project’s OpenPedCan v15 consensus somatic-variant extract, indexed against exact source tumour labels. There are 1,600 Neuroblastoma-labelled biospecimens in the local clinical metadata.

- All 38 target-component symbols occur in the tumour-labelled mutation summary.
- The raw source-row check found 594 gene-associated variant observations across 208 distinct Neuroblastoma-labelled biospecimens (13.0% of the 1,600 labelled biospecimens).
- Those 594 observations include 342 intronic, 89 missense, 20 nonsense, 11 splice-region, 4 splice-site, and 1 in-frame-deletion observations, as well as silent, UTR, and flanking records.
- 43 biospecimens had an observation classified as coding or splice-class; 39 had at least one source-row `HIGH` or `MODERATE` impact annotation. These categories overlap and neither label demonstrates a functional effect.

These are biospecimen-level variant records, not evidence that the complex is activated, inhibited, mutated in a clonal driver, or predictive of drug response. The mutation data and GDSC cell-line assays are not measurements on the same biological samples.

### GDSC2 bortezomib response

The local source manifest identifies the official Wellcome Sanger GDSC2 fitted-dose-response release dated 2023-10-27. The analysis used one bortezomib measurement per model in the indexed response table: 32 models labelled `NB` and 925 other models. Lower `LN_IC50` and lower AUC were treated as the more-sensitive direction.

| Measure | NB median (range) | Other-model median | Median difference (NB − other) | NB lower than a random other model* | One-sided exploratory Mann–Whitney p | 95% bootstrap CI for median difference |
|---|---:|---:|---:|---:|---:|---:|
| `LN_IC50` | −5.234 (−6.178 to −3.941) | −4.920 | −0.314 | 0.572 | 0.083 | −0.523 to 0.217 |
| AUC | 0.759 (0.662 to 0.882) | 0.775 | −0.016 | 0.550 | 0.166 | −0.040 to 0.031 |

*Pairwise probability with ties counted as one half. Bootstrap used 20,000 resamples with seed `20261102`. The two exploratory p-values are both 0.166 after Holm adjustment. The directional tests were not preregistered; treat all statistics as exploratory.

Only 3 of 32 NB models fall in the panel’s most-sensitive decile by `LN_IC50`, and 4 of 32 by AUC. The medians trend toward sensitivity, but the confidence intervals include no difference and the rank comparisons are weak. This does **not** establish Neuroblastoma-selective sensitivity. The comparison pools all other reported cancer types and is not adjusted for lineage or other model covariates.

## Prior art and clinical context

The focused literature/registry review found established preclinical and clinical neuroblastoma work:

- **Preclinical activity:** Brignole et al. studied neuroblastoma cell lines, primary tumour cells, and mouse xenografts; the report described growth inhibition and antitumour effects. [PMID 16912267](https://pubmed.ncbi.nlm.nih.gov/16912267/) · [DOI 10.1093/jnci/djj309](https://doi.org/10.1093/jnci/djj309) (2006).
- **Resistance biology:** Combaret et al. reported resistant neuroblastoma cell lines and increased bortezomib-induced apoptosis when p38 MAPK was inhibited. [PMID 18534018](https://pubmed.ncbi.nlm.nih.gov/18534018/) · [DOI 10.1186/1476-4598-7-50](https://doi.org/10.1186/1476-4598-7-50) (2008).
- **Combination studies:** bortezomib was studied with TRAIL and with all-trans retinoic acid in neuroblastoma models. [PMID 21459798](https://pubmed.ncbi.nlm.nih.gov/21459798/) · [DOI 10.1158/1078-0432.CCR-10-2451](https://doi.org/10.1158/1078-0432.CCR-10-2451) (TRAIL, 2011); [PMID 22087283](https://pubmed.ncbi.nlm.nih.gov/22087283/) · [DOI 10.1371/journal.pone.0027298](https://doi.org/10.1371/journal.pone.0027298) (ATRA, 2011).
- **Combination clinical trial:** Phase I irinotecan plus bortezomib in relapsed/refractory high-risk neuroblastoma, 18 enrolled; the report described two objective responses and modest activity. This is combination evidence, not evidence of bortezomib monotherapy efficacy. [NCT00644696](https://clinicaltrials.gov/study/NCT00644696) · [PMID 28436582](https://pubmed.ncbi.nlm.nih.gov/28436582/) · [DOI 10.1002/pbc.26563](https://doi.org/10.1002/pbc.26563) (2017).
- **DFMO combination trial:** Phase I/II DFMO plus bortezomib, completed; registry results posted 2024, 16 participants enrolled, and Phase II did not proceed. Results are combination results and do not isolate bortezomib’s effect. [NCT02139397](https://clinicaltrials.gov/study/NCT02139397).
- **Mixed paediatric solid-tumour trial:** vorinostat plus bortezomib included neuroblastoma among several registered conditions; it was not a neuroblastoma-specific efficacy trial, and the publication reported no objective responses in the eligible cohort. [NCT01132911](https://clinicaltrials.gov/study/NCT01132911) · [PMID 22887890](https://pubmed.ncbi.nlm.nih.gov/22887890/) · [DOI 10.1002/pbc.24271](https://doi.org/10.1002/pbc.24271) (published online 2012; journal issue 2013).

Thus, prior studies establish that proteasome inhibition and bortezomib combinations have already been investigated in neuroblastoma. The focused search did not find a study explicitly validating this particular 38-component-gene overlap as a biomarker. That is a search result, **not proof of global novelty**; it does not exhaust conference abstracts, unpublished results, or every database.

## Pre-specified expression-score test

I tested one fixed feature: for each of the 38 complex-component genes, transform CCLE RNA-seq TPM as `log2(TPM + 1)`, standardize each gene across all 1,019 cell lines, then take the unweighted mean of the 38 standardized values for each line. Blank TPM cells in the Expression Atlas download were treated as zero. The primary outcome was GDSC2 Bortezomib `LN_IC50`, with AUC as a secondary outcome. Both tests were two-sided; pan-cancer models used reported cancer type as a fixed effect and HC3 robust standard errors. The two outcome p-values were Holm-adjusted.

The expression source is [Expression Atlas E-MTAB-2770](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2770), “RNA-seq of 1019 human cancer cell lines from the CCLE.” The tested gene rows are fingerprinted in the generated summary file.

| Analysis | Models | Estimate per one score unit | 95% HC3 CI | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Pan-cancer, cancer-type-adjusted `LN_IC50` (primary) | 522 | −0.155 | −0.347 to 0.037 | 0.114 | 0.228 |
| Pan-cancer, cancer-type-adjusted AUC (secondary) | 522 | −0.0053 | −0.0172 to 0.0067 | 0.388 | 0.776 |

Neither association is statistically persuasive, and both confidence intervals include no association. In the exploratory Neuroblastoma subset, only 11 of the 32 GDSC2 response models matched an Expression Atlas cell-line name exactly. Spearman correlations were near zero for `LN_IC50` (rho −0.045, p 0.894) and AUC (rho −0.091, p 0.790). This subset is far too small for confirmatory inference.

The join is case-insensitive exact cell-line-name matching, without alias inference. Expression Atlas does not provide the GDSC Sanger model IDs in this matrix, so this is not an independent stable-ID crosswalk. The two datasets are also from different study releases. These limitations, the small Neuroblastoma subset, and the negative results mean this test does **not** support the proposed score as a Bortezomib-response biomarker. It neither disproves every possible proteasome feature nor supports clinical or experimental efficacy.

Reproduce the analysis in an environment with the project requirements installed:

```bash
python scripts/ai/analyze_bortezomib_proteasome_expression.py
```

The script streams only the 38 required rows from Expression Atlas and writes model-level results and a summary to `reports/`.

## Candidate next analysis

### Candidate hypothesis—not yet novel or validated

> **Do pre-specified proteasome-complex molecular features distinguish bortezomib sensitivity among neuroblastoma cell-line models, beyond broad cancer-lineage effects?**

The initial fixed expression-score test did not support this feature. A more useful follow-up would require an independently matched molecular/response dataset and a biologically justified alternative feature—not trying many scores until one appears associated. It should:

1. Obtain an independent, stable-ID crosswalk (Sanger model or COSMIC ID) for any further molecular data; do not broaden the current name-only join with fuzzy aliases.
2. Pre-specify one biologically justified feature before inspecting its association with response. Avoid a 38-predictor model with only 32 NB response models.
3. Seek an additional public drug-screen dataset with Bortezomib measurements and verified model identities before proposing a lab experiment.
4. Re-run a targeted novelty search on the precise feature, model system, and endpoint before describing the question as novel.

### Crosswalk and independent-screen feasibility check (2026-11-02)

The 11 Neuroblastoma models previously joined by exact cell-line name were checked against the [Cellosaurus API](https://api.cellosaurus.org/). Each expression label and GDSC model resolves to the same Cellosaurus accession, and the Cellosaurus cross-reference includes the matching Sanger Cell Model Passport ID. This supports those 11 identity matches with a stable identifier rather than name matching alone.

One additional possible match is GDSC `NB1` (Sanger `SIDM00578`) and the E-MTAB-2770 expression label `NB-1`. Cellosaurus resolves the neuroblastoma record to `CVCL_1440` and `SIDM00578`; an identically named oral-cancer record is distinguishable by its disease annotation. This is a defensible 12th stable-ID match, but the existing score outputs still reflect the prior 11-model exact-name join and must be rerun before reporting updated Neuroblastoma statistics. Related names such as `BE2-M17` and `SK-N-BE(2)` were not merged: Cellosaurus identifies these as distinct lines, despite a derivation relationship.

An independent follow-up candidate is the [Broad Cancer Therapeutics Response Portal (CTRP)](https://portals.broadinstitute.org/ctrp.v2/). Its portal describes a screen of 481 compounds across 860 characterized cancer cell lines, and its compound record lists Bortezomib (compound `411737`, synonym Velcade). A query to the portal's public `aucCorrelationPoints` endpoint returned Bortezomib AUC points for 12 lines annotated to autonomic ganglia. Cellosaurus records identify all 12 as Neuroblastoma and provide stable Cell Model Passport IDs. Eleven of these 12 have matching E-MTAB-2770 CCLE expression samples by Cellosaurus accession; eight also have GDSC2 Bortezomib response records by Sanger model ID.

The 12 CTRP models are CHP126, IMR32, KELLY, KPNSI9S, KPNYN, MHHNB11, NB1, NH6, SIMA, SKNBE2, SKNFI, and SKNSH. NH6 lacks a matched E-MTAB-2770 expression sample. CTRP provides an independent response assay, but the molecular feature values would still come from the same CCLE expression cohort, so this is not a fully independent molecular cohort.

The prespecified 38-gene score was tested against CTRP Bortezomib `cpdAUC` in the 11 models with matched E-MTAB-2770 expression and stable Cellosaurus IDs. The two-sided Spearman correlation was ρ = −0.336 (p = 0.312); a 20,000-resample percentile bootstrap 95% interval was −0.831 to 0.308. Thus, the direction and magnitude remain highly uncertain, and the data do not provide persuasive evidence of an association. CTRP AUC values were identical when retrieved using PSMA1 versus ACTB as the queried expression feature, consistent with the response values being independent of which gene was queried.

The model-level results and analysis summary are saved in `reports/bortezomib_ctrp_proteasome_expression_models.csv` and `reports/bortezomib_ctrp_proteasome_expression_summary.json`. This is an exploratory external-response analysis, not a fully independent molecular validation: the response assay is CTRP, but the expression values remain from the same CCLE E-MTAB-2770 cohort used previously. With only 11 models, it cannot confirm or rule out a biomarker and does not support clinical or predictive claims. Do not select alternative scores based on this small dataset.

**Feasibility decision:** the prespecified external-response check is complete. Further progress should prioritize a larger, independently matched molecular/response cohort or appropriately powered experimental validation, not additional score searches in these 11 models.

## Data gaps and interpretation boundaries

- Expression data are fetched from E-MTAB-2770 at analysis time rather than maintained as a local matched matrix. The original GDSC2 score analysis used exact cell-line-name matching; the CTRP analysis uses audited stable Cellosaurus accessions.
- OpenPedCan tumour variant observations and GDSC2 cell-line response records are different cohorts and cannot be joined into a patient-level response claim.
- The 38-component annotation is a single complex record; the overlap is not 38 independent edges of experimental evidence.
- The present pan-lineage comparison is exploratory and does not provide predictive validation, a clinical efficacy estimate, or a treatment recommendation.
- The original GDSC2 Neuroblastoma expression/response analysis used 11 exact-name matches; one additional stable-ID candidate was identified. CTRP has 11 stable-ID matched expression/response models. Neither small subset supports confirmatory biomarker testing.
- The external novelty review was focused on PubMed/Europe PMC and ClinicalTrials.gov. It cannot certify that an idea has never been tried.

## Reproducibility pointers

- Local evidence index: `data/processed/graph_v3/research_evidence.sqlite`
- Index metadata and source checksums: `data/processed/graph_v3/research_evidence_index_manifest.json`
- OpenPedCan v15-derived mutation input: `data/raw/openpedcan/openpedcan_mutations_phase2.csv`
- GDSC2 input and release metadata: `data/raw/gdsc/gdsc2_responses.csv` and `data/raw/gdsc/gdsc2_manifest.json`
- ChEMBL target extract: `data/raw/chembl/gdsc2_chembl_target_evidence.csv`
- Expression score analysis: `scripts/ai/analyze_bortezomib_proteasome_expression.py`
- Model-level score and response output: `reports/bortezomib_proteasome_expression_models.csv`
- Analysis metadata and results: `reports/bortezomib_proteasome_expression_summary.json`
- Exploratory CTRP analysis: `scripts/ai/analyze_bortezomib_ctrp_expression.py`
- CTRP model-level score and response output: `reports/bortezomib_ctrp_proteasome_expression_models.csv`
- CTRP analysis metadata and results: `reports/bortezomib_ctrp_proteasome_expression_summary.json`
- Structural/index validation command: `./.venv/bin/python scripts/validate_research_evidence_index.py` (passed during this review)
