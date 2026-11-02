from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:  # graceful fallback; install requirements-dashboard.txt for full visuals
    px = None
    go = None
    PLOTLY_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"
GRAPH_V3_DIR = DATA / "processed" / "graph_v3"
RESEARCH_DB = GRAPH_V3_DIR / "research_evidence.sqlite"
RESEARCH_INDEX_MANIFEST = GRAPH_V3_DIR / "research_evidence_index_manifest.json"
GRAPH_V3_MANIFEST = GRAPH_V3_DIR / "graph_v3_manifest.json"
INTERGENIC_DB = GRAPH_V3_DIR / "intergenic_evidence.sqlite"

st.set_page_config(
    page_title="Evidence Atlas | Paediatric Oncology",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system: Evidence Atlas — light scientific workspace, navy navigation,
# teal observed evidence, blue curated mappings, violet derived data, amber
# inference, red data conflict. Labels always accompany colour.
# -----------------------------------------------------------------------------
CSS = r"""
<style>
:root {
  --ea-ink: #14283e;
  --ea-ink-2: #203b56;
  --ea-muted: #64758b;
  --ea-line: #e3e9f0;
  --ea-surface: #ffffff;
  --ea-canvas: #f3f6f9;
  --ea-teal: #13877f;
  --ea-blue: #5279ad;
  --ea-violet: #7958b5;
  --ea-amber: #b97824;
  --ea-red: #b64b52;
}
.stApp { background: var(--ea-canvas); color: var(--ea-ink); }
.block-container { max-width: 1580px; padding-top: 1.25rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #14283e; border-right: 1px solid #263b52; }
[data-testid="stSidebar"] * { color: #e7edf4; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .48rem .5rem; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] hr { border-color: #38516b; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--ea-line); border-radius: 15px; padding: 15px 17px; box-shadow: 0 4px 16px rgba(20,40,62,.035); }
[data-testid="stMetricLabel"] { color: #64758b !important; font-size: .78rem; }
[data-testid="stMetricValue"] { color: #14283e !important; font-weight: 700; letter-spacing: -.04em; }
[data-testid="stMetricDelta"] { font-size: .72rem; }
div[data-testid="stVerticalBlock"] div[data-testid="stAlert"] { border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--ea-line); }
.stTabs [data-baseweb="tab"] { height: 46px; background: transparent; border-radius: 10px 10px 0 0; color: #596b80; }
.stTabs [aria-selected="true"] { color: #14283e !important; border-bottom: 3px solid var(--ea-teal) !important; }
div[data-testid="stForm"] { background: #fff; border: 1px solid var(--ea-line); padding: 1rem; border-radius: 14px; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius: 10px; font-weight: 650; min-height: 2.55rem; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background: #14283e; border-color: #14283e; }
[data-testid="stDataFrame"] { border: 1px solid var(--ea-line); border-radius: 12px; overflow: hidden; }
.ea-brand { display:flex; align-items:center; gap:11px; padding: .3rem .1rem 1rem .1rem; }
.ea-mark { width:39px; height:39px; display:flex; align-items:center; justify-content:center; border-radius:12px; background:linear-gradient(145deg,#35a89e,#146c75); color:white; font-size:19px; font-weight:800; box-shadow:0 5px 15px rgba(19,135,127,.25); }
.ea-brand-title { font-size:15px; font-weight:800; letter-spacing:.12em; line-height:1.1; color:#fff; }
.ea-brand-sub { font-size:9px; letter-spacing:.16em; color:#9eb1c5; margin-top:4px; }
.ea-eyebrow { text-transform:uppercase; color:#6b7e93; font-size:10px; font-weight:800; letter-spacing:.18em; margin-bottom:7px; }
.ea-hero { position:relative; overflow:hidden; background:linear-gradient(112deg,#14283e 0%,#1d3b57 64%,#225e66 100%); color:#f4f8fb; border-radius:20px; padding:27px 30px 26px 30px; margin-bottom:18px; box-shadow:0 12px 28px rgba(20,40,62,.11); }
.ea-hero:after { content:""; position:absolute; right:-65px; top:-130px; width:390px; height:390px; border:1px solid rgba(173,225,220,.22); border-radius:50%; box-shadow:0 0 0 35px rgba(173,225,220,.035),0 0 0 75px rgba(173,225,220,.025),0 0 0 120px rgba(173,225,220,.018); }
.ea-hero h1 { color:#fff; font-size:clamp(27px,3vw,42px); line-height:1.08; letter-spacing:-.045em; font-weight:760; margin:4px 0 10px 0; max-width:850px; }
.ea-hero p { color:#c4d2df; max-width:790px; font-size:14px; line-height:1.55; margin:0; }
.ea-hero .ea-eyebrow { color:#93c9c2; }
.ea-hero-meta { display:flex; gap:9px; flex-wrap:wrap; margin-top:18px; }
.ea-chip { display:inline-flex; align-items:center; border:1px solid rgba(217,237,239,.25); border-radius:999px; padding:5px 10px; font-size:10px; letter-spacing:.04em; color:#e1f0ef; background:rgba(255,255,255,.055); }
.ea-section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin:20px 0 12px 0; }
.ea-section-head h2 { color:var(--ea-ink); font-size:19px; font-weight:760; letter-spacing:-.025em; margin:0; }
.ea-section-head p { color:var(--ea-muted); margin:3px 0 0 0; font-size:11px; }
.ea-card { background:#fff; border:1px solid var(--ea-line); border-radius:15px; padding:17px 18px; box-shadow:0 4px 16px rgba(20,40,62,.035); height:100%; }
.ea-card-label { color:#718198; font-size:10px; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
.ea-card-value { color:#14283e; font-size:27px; letter-spacing:-.045em; font-weight:760; margin-top:7px; line-height:1.05; }
.ea-card-note { color:#738197; font-size:11px; line-height:1.45; margin-top:8px; }
.ea-status { display:inline-flex; align-items:center; gap:6px; padding:5px 9px; font-size:10px; border-radius:999px; font-weight:750; letter-spacing:.035em; }
.ea-status-observed { color:#126b65; background:#e5f4f1; }
.ea-status-curated { color:#345f94; background:#eaf0f8; }
.ea-status-derived { color:#6748a4; background:#f0eafb; }
.ea-status-inferred { color:#8b5c1b; background:#fbf1df; }
.ea-status-unknown { color:#657386; background:#edf0f3; }
.ea-status-conflict { color:#a13c44; background:#fae9ea; }
.ea-panel { background:#fff; border:1px solid var(--ea-line); border-radius:16px; padding:17px 18px; margin:9px 0; }
.ea-panel-title { color:#1a3048; font-size:14px; font-weight:760; margin-bottom:4px; }
.ea-panel-sub { color:#76859a; font-size:11px; line-height:1.45; }
.ea-callout { border-left:3px solid var(--ea-amber); background:#fff9ee; border-radius:0 11px 11px 0; padding:11px 14px; color:#715323; font-size:12px; line-height:1.55; margin:9px 0; }
.ea-callout-teal { border-left-color:var(--ea-teal); background:#edf8f6; color:#245d58; }
.ea-callout-blue { border-left-color:var(--ea-blue); background:#eff4fa; color:#3d5c7d; }
.ea-rule { border:0; border-top:1px solid var(--ea-line); margin:17px 0; }
.ea-step { display:flex; gap:12px; align-items:flex-start; margin:10px 0; }
.ea-step-no { min-width:27px; height:27px; display:flex; justify-content:center; align-items:center; background:#e7f3f1; color:#137b73; border-radius:9px; font-size:11px; font-weight:800; }
.ea-step-title { color:#20384f; font-weight:730; font-size:12px; }
.ea-step-copy { color:#708096; font-size:11px; line-height:1.5; margin-top:3px; }
.ea-footer { color:#8997a8; font-size:10px; padding-top:17px; border-top:1px solid var(--ea-line); margin-top:30px; }
[data-testid="stSidebar"] .ea-sidebar-note { color:#9db0c4; font-size:10px; line-height:1.55; }
@media (max-width: 800px) { .ea-hero { padding:22px; } .ea-hero:after { opacity:.35; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached data access. The dashboard never deserializes the 590 MB graph tensor.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def read_json(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def load_clinical() -> pd.DataFrame:
    path = DATA / "raw/openpedcan/openpedcan_phase1.csv"
    if not path.is_file():
        return pd.DataFrame()
    usecols = ["sample_id", "cohort", "tumor_type", "diagnosis", "molecular_subtype"]
    try:
        return pd.read_csv(path, usecols=lambda c: c in usecols, dtype="string", low_memory=False)
    except (ValueError, OSError, pd.errors.ParserError):
        return pd.DataFrame()


def open_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    uri = "file:" + quote(str(db_path), safe="/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=8)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


@st.cache_data(ttl=300, show_spinner=False)
def index_is_ready(db_path_text: str, manifest_path_text: str) -> tuple[bool, str]:
    db = Path(db_path_text)
    manifest = read_json(manifest_path_text)
    if not db.is_file():
        return False, "Research evidence index is not built yet."
    checks = manifest.get("validation", {})
    if not manifest or not checks or not all(v is True for v in checks.values()):
        return False, "Index manifest is missing or its recorded checks are incomplete."
    return True, "Indexed source evidence is available."


@st.cache_data(ttl=300, show_spinner=False)
def query_frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    with open_readonly(RESEARCH_DB) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=300, show_spinner=False)
def get_drugs() -> pd.DataFrame:
    return query_frame(
        """SELECT drug_id, drug_name, COUNT(*) AS measurement_count
           FROM gdsc_response
           WHERE TRIM(COALESCE(drug_name,'')) <> ''
           GROUP BY drug_id, drug_name
           ORDER BY drug_name COLLATE NOCASE"""
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_gdsc_cancer_types() -> list[str]:
    frame = query_frame(
        """SELECT DISTINCT reported_cancer_type FROM gdsc_response
           WHERE TRIM(COALESCE(reported_cancer_type,'')) <> ''
           ORDER BY reported_cancer_type COLLATE NOCASE"""
    )
    return frame["reported_cancer_type"].dropna().astype(str).tolist()


@st.cache_data(ttl=300, show_spinner=False)
def get_chembl_targets(drug_id: str, drug_name: str) -> pd.DataFrame:
    # Exact GDSC drug ID is preferred; exact case-insensitive name is fallback.
    return query_frame(
        """SELECT gdsc_drug_id, gdsc_drug_name, molecule_chembl_id,
                  drug_match_method, matched_chembl_synonym, target_chembl_id,
                  target_name, target_type, uniprot_accession, gene_symbol,
                  mechanism_of_action, action_type, direct_interaction,
                  max_phase, source_url
           FROM chembl_target_evidence
           WHERE gdsc_drug_id = ?
              OR LOWER(TRIM(gdsc_drug_name)) = LOWER(TRIM(?))
           ORDER BY target_name""",
        (str(drug_id), str(drug_name)),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_disease_gene_summary(tumor_type: str, limit: int = 25) -> pd.DataFrame:
    return query_frame(
        """SELECT gene_symbol, SUM(observation_count) AS observations
           FROM mutation_gene_summary
           WHERE LOWER(TRIM(tumor_type)) = LOWER(TRIM(?))
           GROUP BY gene_symbol
           ORDER BY observations DESC
           LIMIT ?""",
        (tumor_type, int(limit)),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_target_gene_overlap(tumor_type: str, gene_symbols: tuple[str, ...]) -> pd.DataFrame:
    genes = sorted({str(g).strip().upper() for g in gene_symbols if str(g).strip()})
    if not genes:
        return pd.DataFrame(columns=["gene_symbol", "observations"])
    placeholders = ",".join("?" for _ in genes)
    sql = f"""SELECT gene_symbol, SUM(observation_count) AS observations
              FROM mutation_gene_summary
              WHERE LOWER(TRIM(tumor_type)) = LOWER(TRIM(?))
                AND UPPER(TRIM(gene_symbol)) IN ({placeholders})
              GROUP BY gene_symbol
              ORDER BY observations DESC"""
    return query_frame(sql, (tumor_type, *genes))


@st.cache_data(ttl=300, show_spinner=False)
def get_response_measurements(
    drug_id: str, drug_name: str, gdsc_cancer_type: str, max_rows: int = 12000
) -> pd.DataFrame:
    where = ["(drug_id = ? OR LOWER(TRIM(drug_name)) = LOWER(TRIM(?)))"]
    params: list = [str(drug_id), str(drug_name)]
    if gdsc_cancer_type != "All reported cancer types":
        where.append("LOWER(TRIM(reported_cancer_type)) = LOWER(TRIM(?))")
        params.append(gdsc_cancer_type)
    sql = f"""SELECT result_id, drug_id, drug_name, cell_line,
                     sanger_model_id, reported_cancer_type, putative_target,
                     pathway, ln_ic50, auc, rmse
              FROM gdsc_response
              WHERE {' AND '.join(where)}
              ORDER BY result_id
              LIMIT ?"""
    params.append(int(max_rows))
    return query_frame(sql, tuple(params))


@st.cache_data(ttl=300, show_spinner=False)
def get_drug_response_count(drug_id: str, drug_name: str, gdsc_cancer_type: str) -> int:
    where = ["(drug_id = ? OR LOWER(TRIM(drug_name)) = LOWER(TRIM(?)))"]
    params: list = [str(drug_id), str(drug_name)]
    if gdsc_cancer_type != "All reported cancer types":
        where.append("LOWER(TRIM(reported_cancer_type)) = LOWER(TRIM(?))")
        params.append(gdsc_cancer_type)
    frame = query_frame(
        f"SELECT COUNT(*) AS n FROM gdsc_response WHERE {' AND '.join(where)}",
        tuple(params),
    )
    return int(frame.iloc[0]["n"]) if not frame.empty else 0


@st.cache_data(ttl=300, show_spinner=False)
def get_intergenic_sample(sample_id: str, limit: int, offset: int) -> dict:
    with open_readonly(INTERGENIC_DB) as conn:
        summary = conn.execute(
            "SELECT observation_count FROM sample_summary WHERE sample_id=?",
            (sample_id.strip(),),
        ).fetchone()
        if summary is None:
            raise ValueError(f"No intergenic observations found for sample {sample_id!r}.")
        total = int(summary["observation_count"])
        rows = conn.execute(
            """SELECT source_maf_data_row, sample_id, chromosome,
                      start_position, end_position, strand, variant_type,
                      reference_allele, tumor_seq_allele1, tumor_seq_allele2,
                      dbsnp_rs, genome_build, variant_classification
               FROM intergenic_observation
               WHERE sample_id=? ORDER BY source_maf_data_row LIMIT ? OFFSET ?""",
            (sample_id.strip(), int(limit), int(offset)),
        ).fetchall()
    items = [dict(r) for r in rows]
    next_offset = offset + len(items) if offset + len(items) < total else None
    return {"sample_id": sample_id.strip(), "total": total, "items": items,
            "offset": offset, "limit": limit, "next_offset": next_offset}


# -----------------------------------------------------------------------------
# Plot styling helpers
# -----------------------------------------------------------------------------
PALETTE = {
    "navy": "#14283e", "teal": "#13877f", "blue": "#5279ad",
    "violet": "#7958b5", "amber": "#b97824", "red": "#b64b52",
    "muted": "#74849a", "grid": "#e7edf2",
}


def style_figure(fig, height: int = 330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui", color=PALETTE["navy"], size=11),
        margin=dict(l=8, r=12, t=22, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#14283e", font_color="#ffffff", bordercolor="#14283e"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="ea-section-head"><div><h2>{title}</h2><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def status_chip(text: str, category: str = "observed") -> str:
    mapping = {
        "observed": "ea-status-observed", "curated": "ea-status-curated",
        "derived": "ea-status-derived", "inferred": "ea-status-inferred",
        "unknown": "ea-status-unknown", "conflict": "ea-status-conflict",
    }
    cls = mapping.get(category, "ea-status-unknown")
    return f'<span class="ea-status {cls}">{text}</span>'


def metric_card(label: str, value: str, note: str, colour: str = "teal") -> None:
    st.markdown(
        f'<div class="ea-card"><div class="ea-card-label">{label}</div>'
        f'<div class="ea-card-value">{value}</div><div class="ea-card-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def make_figure_unavailable(message: str) -> None:
    st.info(message)


# -----------------------------------------------------------------------------
# Load metadata — no large graph tensor is loaded by this dashboard.
# -----------------------------------------------------------------------------
clinical = load_clinical()
graph_manifest = read_json(str(GRAPH_V3_MANIFEST))
mutation_manifest = read_json(str(DATA / "raw/openpedcan/openpedcan_mutations_manifest.json"))
gdsc_manifest = read_json(str(DATA / "raw/gdsc/gdsc2_manifest.json"))
chembl_manifest = read_json(str(DATA / "raw/chembl/gdsc2_chembl_manifest.json"))
index_manifest = read_json(str(RESEARCH_INDEX_MANIFEST))
research_ready, research_message = index_is_ready(str(RESEARCH_DB), str(RESEARCH_INDEX_MANIFEST))

clinical_count = int(clinical["sample_id"].nunique()) if not clinical.empty and "sample_id" in clinical else 0
mutation_count = int(mutation_manifest.get("variant_rows_matched", 0) or 0)
if not mutation_count:
    mutation_count = int(index_manifest.get("counts", {}).get("mutation_rows_scanned", 0) or 0)
intergenic_count = int(graph_manifest.get("new_node_count", 0) or 0)
intergenic_sample_count = int(graph_manifest.get("sample_nodes_matched", 0) or 0)
gdsc_count = int(gdsc_manifest.get("response_rows", 0) or 0)
chembl_count = int(chembl_manifest.get("mechanism_gene_rows", 0) or 0)

# -----------------------------------------------------------------------------
# Sidebar/navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="ea-brand"><div class="ea-mark">E</div><div>'
        '<div class="ea-brand-title">EVIDENCE ATLAS</div>'
        '<div class="ea-brand-sub">PAEDIATRIC ONCOLOGY INTELLIGENCE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ea-sidebar-note">A traceable research environment connecting clinical context, molecular observations and experimental evidence.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='ea-eyebrow' style='color:#8fa6bc'>WORKSPACES</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "Command centre",
            "Drug–disease investigation",
            "Evidence relationship map",
            "Sample explorer",
            "Provenance & quality",
            "Investor briefing",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="ea-eyebrow" style="color:#8fa6bc">PLATFORM STATE</div>', unsafe_allow_html=True)
    if graph_manifest and (GRAPH_V3_DIR / "typed_graph_v3.pt").is_file():
        st.markdown(status_chip("GRAPH V3 ARTIFACT PRESENT", "observed"), unsafe_allow_html=True)
    else:
        st.markdown(status_chip("GRAPH V3 NOT FOUND", "unknown"), unsafe_allow_html=True)
    if research_ready:
        st.markdown(status_chip("RESEARCH INDEX READY", "curated"), unsafe_allow_html=True)
    else:
        st.markdown(status_chip("RESEARCH INDEX REQUIRED", "inferred"), unsafe_allow_html=True)
    st.markdown("<div class='ea-sidebar-note'>Prediction capability is not enabled. This interface presents research evidence, not treatment recommendations.</div>", unsafe_allow_html=True)

# Common hero banner.
page_heroes = {
    "Command centre": ("RESEARCH OVERVIEW", "Evidence, connected with context.", "A live view of the platform's current research assets, integration status and evidence coverage."),
    "Drug–disease investigation": ("INVESTIGATION WORKSPACE", "Interrogate the evidence behind a drug–disease hypothesis.", "Separate target records, tumour molecular context and in-vitro response observations. No composite efficacy score is generated."),
    "Evidence relationship map": ("RELATIONSHIP INTELLIGENCE", "See what is connected — and what is not.", "Explore source-supported relationships while keeping target annotations, tumour observations and assay measurements distinct."),
    "Sample explorer": ("SAMPLE-LEVEL EVIDENCE", "Move from cohort context to individual observations.", "Inspect intergenic records with source-row provenance and bounded pagination."),
    "Provenance & quality": ("DATA INTEGRITY", "Every observation should have a history.", "Inspect source releases, checksums, coordinate conventions and validation boundaries."),
    "Investor briefing": ("PLATFORM THESIS", "A paediatric oncology evidence intelligence layer.", "A clear view of what is built, what is differentiated and what remains to be demonstrated."),
}
hero_kicker, hero_title, hero_copy = page_heroes[page]
st.markdown(
    f'<div class="ea-hero"><div class="ea-eyebrow">{hero_kicker}</div>'
    f'<h1>{hero_title}</h1><p>{hero_copy}</p>'
    '<div class="ea-hero-meta"><span class="ea-chip">PAEDIATRIC-FIRST CONTEXT</span>'
    '<span class="ea-chip">SOURCE-TRACEABLE EVIDENCE</span>'
    '<span class="ea-chip">REPRODUCIBLE RESEARCH</span></div></div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Page 1: Command centre
# -----------------------------------------------------------------------------
if page == "Command centre":
    section("Platform at a glance", "Dataset-level counts. Each metric has its own source population and definition.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Clinical biospecimens", f"{clinical_count:,}" if clinical_count else "—", "Unique sample IDs in OpenPedCan clinical metadata", "blue")
    with c2: metric_card("Mutation observations", f"{mutation_count:,}" if mutation_count else "—", "Matched mutation records; includes intergenic records in the source count", "violet")
    with c3: metric_card("Intergenic observations", f"{intergenic_count:,}" if intergenic_count else "—", f"{intergenic_sample_count:,} sample barcodes represented in Graph V3", "teal")
    with c4: metric_card("GDSC2 response records", f"{gdsc_count:,}" if gdsc_count else "—", "In-vitro cell-line measurements; not clinical outcomes", "amber")

    if not research_ready:
        st.markdown(
            '<div class="ea-callout">The compact research index has not been built yet. The overview remains available, but disease-specific gene overlap and response distributions require the index.</div>',
            unsafe_allow_html=True,
        )
        st.code("./.venv/bin/python scripts/build_research_evidence_index.py", language="bash")

    section("Cohort landscape", "OpenPedCan clinical metadata — sample counts by reported tumour type.")
    if not clinical.empty and "tumor_type" in clinical:
        cohort_counts = (
            clinical.dropna(subset=["tumor_type"])
            .groupby("tumor_type")["sample_id"].nunique()
            .sort_values(ascending=False).head(14).reset_index(name="biospecimens")
        )
        left, right = st.columns([1.35, 1])
        with left:
            if PLOTLY_AVAILABLE and not cohort_counts.empty:
                fig = px.bar(
                    cohort_counts.sort_values("biospecimens"),
                    x="biospecimens", y="tumor_type", orientation="h",
                    color_discrete_sequence=[PALETTE["teal"]],
                    labels={"biospecimens": "Unique biospecimens", "tumor_type": "Tumour type"},
                )
                style_figure(fig, 390)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.bar_chart(cohort_counts.set_index("tumor_type"))
        with right:
            st.markdown('<div class="ea-panel"><div class="ea-panel-title">Evidence domains</div><div class="ea-panel-sub">Current data availability, not a measure of scientific completeness.</div><hr class="ea-rule">', unsafe_allow_html=True)
            domain_rows = [
                ("Clinical context", clinical_count, "Observed", "observed"),
                ("Molecular variants", mutation_count, "Source indexed", "derived"),
                ("Intergenic observations", intergenic_count, "Graph integrated", "derived"),
                ("Drug-target annotations", chembl_count, "ChEMBL extract", "curated"),
                ("Cell-line response", gdsc_count, "GDSC2 assays", "observed"),
            ]
            for label, value, note, kind in domain_rows:
                st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;margin:10px 0'><div><b style='font-size:12px;color:#20384f'>{label}</b><div style='font-size:10px;color:#7b899a'>{note}</div></div><div style='text-align:right'><b style='color:#20384f'>{value:,}</b><br>{status_chip(kind.upper(), kind)}</div></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("OpenPedCan clinical metadata was not found at the expected path.")

    section("Research readiness", "What the current data supports — and what still requires validation.")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="ea-card"><div class="ea-card-label">01 · Target engagement</div><div class="ea-card-value" style="font-size:20px">Source annotations available</div><div class="ea-card-note">ChEMBL drug matching and mechanism records are present. These require record-level review and do not prove activity in a paediatric tumour.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="ea-card"><div class="ea-card-label">02 · Disease context</div><div class="ea-card-value" style="font-size:20px">Molecular context integrated</div><div class="ea-card-note">OpenPedCan supplies tumour-associated observations. Exact target-gene overlap is descriptive, not proof of dependency or target essentiality.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="ea-card"><div class="ea-card-label">03 · Response evidence</div><div class="ea-card-value" style="font-size:20px">In-vitro assays available</div><div class="ea-card-note">GDSC2 measurements can be explored. A validated paediatric response predictor and clinical efficacy claim are not available.</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page 2: Drug-disease investigation
# -----------------------------------------------------------------------------
elif page == "Drug–disease investigation":
    if not research_ready:
        st.warning("Build the research evidence index before opening this workspace.")
        st.code("./.venv/bin/python scripts/build_research_evidence_index.py", language="bash")
    else:
        try:
            drugs = get_drugs()
            cancer_types = get_gdsc_cancer_types()
        except (sqlite3.Error, OSError) as exc:
            drugs, cancer_types = pd.DataFrame(), []
            st.error(f"Could not read the research index: {exc}")

        if drugs.empty:
            st.warning("No GDSC2 drugs are available in the research index.")
        else:
            # Use a two-column selector band and preserve the origin label.
            drug_options = drugs.drop_duplicates(subset=["drug_id", "drug_name"]).copy()
            drug_options["label"] = drug_options.apply(
                lambda r: f"{r['drug_name']}  ·  GDSC ID {r['drug_id']}", axis=1
            )
            disease_options = []
            if not clinical.empty and "tumor_type" in clinical:
                disease_options = sorted(clinical["tumor_type"].dropna().astype(str).unique().tolist())
            if not disease_options:
                disease_options = ["Not available"]
            left, right = st.columns(2)
            with left:
                drug_label = st.selectbox("Investigate drug", drug_options["label"].tolist(), index=0)
            selected_drug = drug_options.loc[drug_options["label"] == drug_label].iloc[0]
            with right:
                selected_disease = st.selectbox("Paediatric tumour context", disease_options)

            st.markdown('<div class="ea-callout ea-callout-blue">This is an evidence investigation, not a therapeutic recommendation. Drug identifiers, tumour labels and model systems are shown with their source context. Exact string overlap is not treated as proof of biological equivalence.</div>', unsafe_allow_html=True)

            target_records = get_chembl_targets(str(selected_drug["drug_id"]), str(selected_drug["drug_name"]))
            gene_values = []
            if not target_records.empty and "gene_symbol" in target_records:
                for value in target_records["gene_symbol"].dropna().astype(str):
                    gene_values.extend([g.strip() for g in value.replace(";", ",").split(",") if g.strip()])
            target_genes = tuple(sorted({g.upper() for g in gene_values if g.upper() not in {"", "NAN", "UNKNOWN"}}))

            tumor_sample_count = 0
            if not clinical.empty and "tumor_type" in clinical:
                match = clinical[clinical["tumor_type"].astype(str).str.casefold() == str(selected_disease).casefold()]
                tumor_sample_count = int(match["sample_id"].nunique()) if "sample_id" in match else len(match)

            overlap = get_target_gene_overlap(selected_disease, target_genes) if target_genes else pd.DataFrame()
            disease_top = get_disease_gene_summary(selected_disease, 15)

            # Three question panels.
            section("Three questions. Three evidence layers.", "Each panel is sourced independently; no composite confidence or efficacy score is calculated.")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.markdown('<div class="ea-card"><div class="ea-card-label">01 · What does it hit?</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ea-card-value">{target_records["target_chembl_id"].nunique() if not target_records.empty and "target_chembl_id" in target_records else 0:,}</div><div class="ea-card-note">Distinct ChEMBL target IDs in the exact drug-mapping extract.</div></div>', unsafe_allow_html=True)
            with q2:
                st.markdown('<div class="ea-card"><div class="ea-card-label">02 · Why this disease?</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ea-card-value">{len(overlap):,}</div><div class="ea-card-note">Target genes with gene-associated variant observations in the selected OpenPedCan tumour label.</div></div>', unsafe_allow_html=True)
            with q3:
                response_count = get_drug_response_count(str(selected_drug["drug_id"]), str(selected_drug["drug_name"]), "All reported cancer types")
                st.markdown('<div class="ea-card"><div class="ea-card-label">03 · Does it look like it will work?</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ea-card-value">{response_count:,}</div><div class="ea-card-note">GDSC2 in-vitro response measurements for this drug ID/name, across reported cancer models.</div></div>', unsafe_allow_html=True)

            section("01 · Target engagement", "ChEMBL annotations for the selected GDSC drug. Matching method remains visible.")
            if target_records.empty:
                st.info("No matching ChEMBL target evidence was found for this exact GDSC drug ID/name in the current extract. This is absence from the current extract, not proof that the drug has no molecular targets.")
            else:
                target_cols = [c for c in ["target_name", "gene_symbol", "target_type", "mechanism_of_action", "action_type", "direct_interaction", "drug_match_method", "molecule_chembl_id", "target_chembl_id", "max_phase", "source_url"] if c in target_records]
                st.dataframe(target_records[target_cols], use_container_width=True, hide_index=True, height=270)
                if target_genes:
                    st.caption("Gene symbols extracted from ChEMBL rows: " + ", ".join(target_genes))
                else:
                    st.warning("ChEMBL rows are present, but no gene symbols were available to perform an exact gene-level overlap with tumour observations.")
                st.markdown('<div class="ea-callout">A ChEMBL mechanism or target annotation establishes what the source records report about the compound. It does not by itself establish target engagement in paediatric tumour tissue or therapeutic efficacy.</div>', unsafe_allow_html=True)

            section("02 · Why this disease?", f"OpenPedCan tumour context: {selected_disease} · {tumor_sample_count:,} unique biospecimens in the clinical metadata.")
            if not target_genes:
                st.info("Disease-specific target overlap is unavailable because the selected drug has no gene-resolved ChEMBL target records in the current extract.")
            elif overlap.empty:
                st.warning("No exact target-gene overlap was found in the indexed gene-associated variant observations for this tumour label. The current analysis does not assess expression, dependency, regulatory effects or pathway causality.")
            else:
                lcol, rcol = st.columns([1.1, 1])
                with lcol:
                    st.markdown('<div class="ea-panel-title">Target genes observed in the selected tumour context</div>', unsafe_allow_html=True)
                    st.dataframe(overlap, use_container_width=True, hide_index=True, height=260)
                with rcol:
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            overlap.sort_values("observations"),
                            x="observations", y="gene_symbol", orientation="h",
                            color_discrete_sequence=[PALETTE["violet"]],
                            labels={"observations": "Variant observations", "gene_symbol": "Target gene"},
                        )
                        style_figure(fig, 280)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.bar_chart(overlap.set_index("gene_symbol"))
                st.markdown('<div class="ea-callout">Interpretation boundary: an observed gene-associated variant is not evidence that the tumour depends on that gene, that the variant activates or inhibits the target, or that the selected drug will affect the tumour. Functional, expression, dependency and disease-specific experimental evidence are still needed.</div>', unsafe_allow_html=True)

            if not disease_top.empty:
                with st.expander("Explore the broader molecular context of this tumour label"):
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            disease_top.sort_values("observations"),
                            x="observations", y="gene_symbol", orientation="h",
                            color_discrete_sequence=[PALETTE["blue"]],
                            labels={"observations": "Variant observations", "gene_symbol": "Gene"},
                        )
                        style_figure(fig, 410)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.dataframe(disease_top, use_container_width=True)
                    st.caption("Counts are source-record observations, not unique patients, pathogenicity assessments or target-dependency measurements. Intergenic IGR/Unknown records are excluded from this gene-associated summary.")

            section("03 · Does it look like it will work?", "Descriptive GDSC2 cell-line assay evidence — not a paediatric efficacy prediction.")
            gdsc_options = ["All reported cancer types", *cancer_types]
            selected_gdsc_type = st.selectbox(
                "GDSC2 reported cancer type (kept separate from the OpenPedCan tumour label)",
                gdsc_options,
                index=0,
                help="The GDSC2 label is not automatically mapped to the selected OpenPedCan tumour type.",
            )
            response_count_filtered = get_drug_response_count(str(selected_drug["drug_id"]), str(selected_drug["drug_name"]), selected_gdsc_type)
            response = get_response_measurements(str(selected_drug["drug_id"]), str(selected_drug["drug_name"]), selected_gdsc_type)
            if response.empty:
                st.info("No GDSC2 response measurements were found for this drug and selected GDSC2 cancer label.")
            else:
                response["ln_ic50"] = pd.to_numeric(response["ln_ic50"], errors="coerce")
                response["auc"] = pd.to_numeric(response["auc"], errors="coerce")
                response["rmse"] = pd.to_numeric(response["rmse"], errors="coerce")
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("Measurements in selected filter", f"{response_count_filtered:,}")
                with m2: st.metric("Cell lines represented", f"{response['cell_line'].nunique():,}")
                with m3: st.metric("Median ln(IC50)", f"{response['ln_ic50'].median():.3f}" if response["ln_ic50"].notna().any() else "—")
                with m4: st.metric("Median AUC", f"{response['auc'].median():.3f}" if response["auc"].notna().any() else "—")
                p1, p2 = st.columns(2)
                with p1:
                    if PLOTLY_AVAILABLE and response["ln_ic50"].notna().any():
                        fig = px.histogram(response.dropna(subset=["ln_ic50"]), x="ln_ic50", nbins=28, color_discrete_sequence=[PALETTE["teal"]], labels={"ln_ic50": "ln(IC50)"})
                        style_figure(fig, 300)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("ln(IC50) distribution unavailable for this filter.")
                with p2:
                    if PLOTLY_AVAILABLE and response["auc"].notna().any():
                        fig = px.histogram(response.dropna(subset=["auc"]), x="auc", nbins=28, color_discrete_sequence=[PALETTE["violet"]], labels={"auc": "AUC"})
                        style_figure(fig, 300)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("AUC distribution unavailable for this filter.")
                display_cols = [c for c in ["cell_line", "sanger_model_id", "reported_cancer_type", "ln_ic50", "auc", "rmse", "putative_target", "pathway"] if c in response]
                with st.expander("Inspect underlying cell-line measurements"):
                    st.dataframe(response[display_cols].head(300), use_container_width=True, hide_index=True, height=300)
                st.markdown('<div class="ea-callout">These are in-vitro cell-line assay measurements from GDSC2. They do not establish clinical response, patient benefit, paediatric tumour efficacy or probability of success. The distribution is descriptive and is not a trained prediction.</div>', unsafe_allow_html=True)

            section("Evidence completeness — no composite score", "A transparent checklist is more honest than a single speculative confidence number.")
            checks = [
                ("Drug-to-target source records", not target_records.empty, "ChEMBL extract"),
                ("Gene-resolved target mapping", bool(target_genes), "ChEMBL gene symbols"),
                ("Tumour-context gene observations", not disease_top.empty, "OpenPedCan mutation summary"),
                ("Exact target-gene overlap", not overlap.empty, "Exact gene-symbol match"),
                ("In-vitro response measurements", response_count > 0, "GDSC2"),
                ("Independent paediatric response validation", False, "Not established in current build"),
                ("Clinical efficacy evidence for this pairing", False, "Not established in current build"),
            ]
            for label, passed, source in checks:
                icon = "✓" if passed else "—"
                colour = "#13877f" if passed else "#8a96a6"
                st.markdown(f"<div style='display:flex;justify-content:space-between;border-bottom:1px solid #e7edf2;padding:9px 2px;gap:12px'><div><b style='font-size:12px;color:#20384f'>{label}</b><div style='font-size:10px;color:#8190a1'>{source}</div></div><div style='font-weight:800;color:{colour}'>{icon}</div></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page 3: Evidence relationship map — only source-backed links are drawn.
# -----------------------------------------------------------------------------
elif page == "Evidence relationship map":
    st.markdown('<div class="ea-callout ea-callout-blue">The map below is assembled from the selected drug’s ChEMBL target records, exact gene-symbol overlap with OpenPedCan mutation summaries, and its GDSC2 assay records. Dashed links indicate observational overlap only, not causality.</div>', unsafe_allow_html=True)
    if not research_ready:
        st.warning("Build the research evidence index to enable this view.")
    else:
        drugs = get_drugs()
        if drugs.empty:
            st.warning("No GDSC2 drug records are indexed.")
        else:
            options = drugs.drop_duplicates(subset=["drug_id", "drug_name"]).copy()
            options["label"] = options.apply(lambda r: f"{r['drug_name']} · {r['drug_id']}", axis=1)
            selected_label = st.selectbox("Drug for relationship exploration", options["label"].tolist())
            selected = options.loc[options["label"] == selected_label].iloc[0]
            diseases = sorted(clinical["tumor_type"].dropna().astype(str).unique().tolist()) if not clinical.empty and "tumor_type" in clinical else []
            disease = st.selectbox("Tumour context", diseases) if diseases else ""
            targets = get_chembl_targets(str(selected["drug_id"]), str(selected["drug_name"]))
            genes = []
            if not targets.empty:
                for value in targets["gene_symbol"].dropna().astype(str):
                    genes.extend([v.strip() for v in value.replace(";", ",").split(",") if v.strip()])
            genes = sorted({g.upper() for g in genes if g.upper() not in {"", "UNKNOWN", "NAN"}})
            overlap = get_target_gene_overlap(disease, tuple(genes)) if disease and genes else pd.DataFrame()
            responses_n = get_drug_response_count(str(selected["drug_id"]), str(selected["drug_name"]), "All reported cancer types")

            if PLOTLY_AVAILABLE and not targets.empty:
                # Layered evidence map. Only edges that exist in the source
                # extract are drawn. The dashed final edge is merely a
                # gene-observation overlap, labelled as such.
                rows = targets.head(18).to_dict("records")
                nodes: dict[str, dict] = {}
                edges: list[tuple[str, str, str, str]] = []
                drug_node = f"drug:{selected['drug_id']}"
                nodes[drug_node] = {"label": str(selected["drug_name"]), "kind": "Drug", "x": 0.04, "y": 0.52}
                molecule_ids = sorted({str(r.get("molecule_chembl_id") or "Unresolved molecule") for r in rows})
                target_ids = sorted({str(r.get("target_chembl_id") or r.get("target_name") or "Unresolved target") for r in rows})
                for i, molecule in enumerate(molecule_ids):
                    key = f"mol:{molecule}"
                    nodes[key] = {"label": molecule, "kind": "ChEMBL molecule", "x": 0.27, "y": 0.88 - i * (0.72 / max(1, len(molecule_ids)-1))}
                    edges.append((drug_node, key, "drug-to-molecule mapping", "solid"))
                target_y: dict[str, float] = {}
                for i, target in enumerate(target_ids):
                    key = f"target:{target}"
                    y = 0.9 - i * (0.78 / max(1, len(target_ids)-1))
                    target_y[target] = y
                    label_row = next((r for r in rows if str(r.get("target_chembl_id") or r.get("target_name") or "Unresolved target") == target), {})
                    nodes[key] = {"label": str(label_row.get("target_name") or target), "kind": "Target", "x": 0.58, "y": y, "id": target}
                for r in rows:
                    molecule = str(r.get("molecule_chembl_id") or "Unresolved molecule")
                    target = str(r.get("target_chembl_id") or r.get("target_name") or "Unresolved target")
                    edges.append((f"mol:{molecule}", f"target:{target}", "ChEMBL target evidence", "solid"))
                overlap_genes = set(overlap["gene_symbol"].astype(str).str.upper()) if not overlap.empty else set()
                gene_ids = sorted({str(r.get("gene_symbol") or "").strip().upper() for r in rows if str(r.get("gene_symbol") or "").strip() and str(r.get("gene_symbol")).upper() != "UNKNOWN"})
                for i, gene in enumerate(gene_ids):
                    key = f"gene:{gene}"
                    y = 0.9 - i * (0.78 / max(1, len(gene_ids)-1))
                    nodes[key] = {"label": gene, "kind": "Gene symbol", "x": 0.82, "y": y}
                    matching_targets = [str(r.get("target_chembl_id") or r.get("target_name") or "Unresolved target") for r in rows if str(r.get("gene_symbol") or "").strip().upper() == gene]
                    for target in matching_targets:
                        edges.append((f"target:{target}", key, "target-to-gene annotation", "solid"))
                    if gene in overlap_genes:
                        disease_key = f"disease:{disease}"
                        nodes[disease_key] = {"label": f"{disease}\nvariant observations", "kind": "Tumour context", "x": 0.98, "y": 0.52}
                        edges.append((key, disease_key, "observed gene-variant overlap · not causal", "dash"))

                edge_x, edge_y = [], []
                for source, target, label, dash in edges:
                    if source in nodes and target in nodes:
                        edge_x.extend([nodes[source]["x"], nodes[target]["x"], None])
                        edge_y.extend([nodes[source]["y"], nodes[target]["y"], None])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#9aabba", width=1.5), hoverinfo="skip", showlegend=False))
                kind_colors = {"Drug": PALETTE["navy"], "ChEMBL molecule": PALETTE["blue"], "Target": PALETTE["teal"], "Gene symbol": PALETTE["violet"], "Tumour context": PALETTE["amber"]}
                for kind in kind_colors:
                    group = [(key, value) for key, value in nodes.items() if value["kind"] == kind]
                    if not group:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[v["x"] for _, v in group], y=[v["y"] for _, v in group],
                        mode="markers+text", text=[v["label"] for _, v in group],
                        textposition="bottom center", textfont=dict(size=9, color=PALETTE["navy"]),
                        marker=dict(size=17 if kind in {"Drug", "Tumour context"} else 12, color=kind_colors[kind], line=dict(color="#ffffff", width=1.5)),
                        name=kind, customdata=[v.get("id", key) for key, v in group],
                        hovertemplate="%{text}<br>Class: " + kind + "<extra></extra>"
                    ))
                fig.update_layout(
                    height=570, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                    margin=dict(l=10,r=10,t=15,b=20), font=dict(color=PALETTE["navy"]),
                    xaxis=dict(visible=False, range=[-0.03,1.08]), yaxis=dict(visible=False, range=[-0.04,1.05]),
                    legend=dict(orientation="h", y=1.08), hovermode="closest",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption("Solid links: recorded ChEMBL drug/molecule/target/gene annotations. Dashed links: exact gene-symbol overlap with gene-associated variant observations in the selected OpenPedCan tumour label. No causal or therapeutic relationship is implied.")
            elif targets.empty:
                st.info("No ChEMBL target chain is available for this exact selected drug mapping. The map will not invent missing connections.")
            else:
                st.dataframe(targets, use_container_width=True)

            st.markdown('<div class="ea-panel"><div class="ea-panel-title">Separate experimental evidence lane</div><div class="ea-panel-sub">GDSC2 assay measurements for the selected drug: '+f"{responses_n:,}"+' records across the indexed cell-line dataset. These data are shown separately from patient molecular observations because the two evidence types do not form a direct sample-level causal link.</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page 4: Sample explorer
# -----------------------------------------------------------------------------
elif page == "Sample explorer":
    section("Intergenic observation browser", "Search sample barcodes and inspect a bounded page of source-linked observations.")
    if not INTERGENIC_DB.is_file():
        st.warning("Intergenic SQLite index was not found. Build it before using sample exploration.")
    else:
        if "ea_sample_id" not in st.session_state:
            st.session_state.ea_sample_id = ""
        if "ea_sample_offset" not in st.session_state:
            st.session_state.ea_sample_offset = 0
        if "ea_sample_limit" not in st.session_state:
            st.session_state.ea_sample_limit = 50
        with st.form("ea_sample_search"):
            sample_input = st.text_input("OpenPedCan sample barcode", value=st.session_state.ea_sample_id, placeholder="e.g. BS_ZSH09N84")
            page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=[25,50,100,200].index(st.session_state.ea_sample_limit))
            submitted = st.form_submit_button("Open sample evidence", type="primary")
        if submitted:
            st.session_state.ea_sample_id = sample_input.strip()
            st.session_state.ea_sample_limit = int(page_size)
            st.session_state.ea_sample_offset = 0
        if st.session_state.ea_sample_id:
            try:
                result = get_intergenic_sample(
                    st.session_state.ea_sample_id,
                    st.session_state.ea_sample_limit,
                    st.session_state.ea_sample_offset,
                )
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Sample observations", f"{result['total']:,}")
                with m2: st.metric("Displayed on this page", f"{len(result['items']):,}")
                with m3: st.metric("Current source-row offset", f"{result['offset']:,}")
                rows = pd.DataFrame(result["items"])
                if not rows.empty:
                    st.dataframe(rows, use_container_width=True, hide_index=True, height=430)
                    if PLOTLY_AVAILABLE:
                        chromosome = rows.groupby("chromosome").size().reset_index(name="observations")
                        fig = px.bar(chromosome, x="chromosome", y="observations", color_discrete_sequence=[PALETTE["teal"]], labels={"chromosome":"Chromosome", "observations":"Rows on current page"})
                        style_figure(fig, 240)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                left, right, note = st.columns([1,1,2])
                with left:
                    if st.button("← Previous", disabled=result["offset"] == 0, use_container_width=True):
                        st.session_state.ea_sample_offset = max(0, result["offset"] - result["limit"])
                        st.rerun()
                with right:
                    if st.button("Next →", disabled=result["next_offset"] is None, use_container_width=True):
                        st.session_state.ea_sample_offset = result["next_offset"]
                        st.rerun()
                with note:
                    st.caption(f"Showing {result['offset'] + 1:,}–{result['offset'] + len(result['items']):,} of {result['total']:,} · GRCh38 · 1-based inclusive")
                st.markdown('<div class="ea-callout">These records are classified as intergenic observations. This view does not assign nearby genes, regulatory effects or functional consequences.</div>', unsafe_allow_html=True)
            except ValueError as exc:
                st.warning(str(exc))
            except (OSError, sqlite3.Error) as exc:
                st.error(f"Could not query the intergenic evidence index: {exc}")
        else:
            st.info("Enter a sample barcode to open its intergenic evidence profile.")

# -----------------------------------------------------------------------------
# Page 5: Provenance and quality centre
# -----------------------------------------------------------------------------
elif page == "Provenance & quality":
    section("Source registry", "The source files and manifests currently available in the project.")
    sources = [
        ("OpenPedCan clinical", DATA / "raw/openpedcan/openpedcan_phase1.csv", "Clinical sample and tumour metadata", "Observed"),
        ("OpenPedCan mutations", DATA / "raw/openpedcan/openpedcan_mutations_phase2.csv", "Consensus molecular variant observations", "Observed"),
        ("OpenPedCan intergenic extract", DATA / "processed/graph_v2/intergenic_variants_maf.csv.gz", "GRCh38 IGR observations and source-row locator", "Derived extract"),
        ("GDSC2 response", DATA / "raw/gdsc/gdsc2_responses.csv", "In-vitro cell-line response measurements", "Observed"),
        ("ChEMBL target evidence", DATA / "raw/chembl/gdsc2_chembl_target_evidence.csv", "Drug-to-molecule/target annotations", "Curated extract"),
        ("Graph V3", GRAPH_V3_DIR / "typed_graph_v3.pt", "Typed graph artifact", "Derived graph"),
        ("Intergenic evidence index", INTERGENIC_DB, "Sample-indexed evidence lookup", "Derived index"),
        ("Research evidence index", RESEARCH_DB, "Disease/gene and response lookup index", "Derived index"),
    ]
    source_table = []
    for name, path, desc, kind in sources:
        source_table.append({
            "Evidence asset": name,
            "Description": desc,
            "Classification": kind,
            "Status": "Present" if path.is_file() else "Not found",
            "Size (MiB)": round(path.stat().st_size / (1024**2), 2) if path.is_file() else None,
        })
    st.dataframe(pd.DataFrame(source_table), use_container_width=True, hide_index=True)

    section("Graph V3 build record", "Manifest-recorded checks are displayed as recorded; this screen does not rerun the full graph validator.")
    if graph_manifest:
        left, right = st.columns(2)
        with left:
            st.metric("Intergenic observation nodes", f"{graph_manifest.get('new_node_count', 0):,}")
            st.metric("Matched sample barcodes", f"{graph_manifest.get('sample_nodes_matched', 0):,}")
        with right:
            st.metric("Original node types preserved", len(graph_manifest.get("original_node_counts_preserved", {})))
            st.metric("Original relation types preserved", len(graph_manifest.get("original_edge_counts_preserved", {})))
        st.markdown(f"**Genome reference:** {graph_manifest.get('coordinate_reference', 'Not recorded')} · **Coordinates:** {graph_manifest.get('coordinate_convention', 'Not recorded')}")
        checks = graph_manifest.get("validation", {})
        for name, value in checks.items():
            st.markdown(f"{status_chip('PASS' if value is True else 'REVIEW', 'observed' if value is True else 'conflict')} &nbsp; {name.replace('_', ' ')}", unsafe_allow_html=True)
    else:
        st.warning("Graph V3 manifest not found.")

    section("Research index provenance", "Checksums bind the index build to specific source-file versions.")
    if index_manifest:
        st.json({
            "schema_version": index_manifest.get("schema_version"),
            "source_hashes_sha256": index_manifest.get("source_hashes_sha256"),
            "counts": index_manifest.get("counts"),
            "summary_rows": index_manifest.get("summary_rows"),
            "mutation_policy": index_manifest.get("mutation_policy"),
            "response_policy": index_manifest.get("response_policy"),
            "validation": index_manifest.get("validation"),
        })
    else:
        st.info("Research evidence index manifest is not available. Build the index to establish its source checksums and row counts.")

# -----------------------------------------------------------------------------
# Page 6: Investor briefing — truthful, stage-aware narrative
# -----------------------------------------------------------------------------
elif page == "Investor briefing":
    section("The thesis", "A specialised evidence layer for paediatric oncology research.")
    st.markdown('<div class="ea-panel"><div class="ea-eyebrow">PRODUCT POSITIONING</div><div style="font-size:27px;font-weight:780;letter-spacing:-.04em;color:#14283e;line-height:1.12">Not another cancer data portal.<br>An evidence intelligence layer.</div><p style="font-size:13px;color:#66778b;line-height:1.65;margin-top:12px">Connect clinical context, molecular observations and experimental evidence while preserving source lineage, evidence semantics and uncertainty. Make research reasoning inspectable and reproducible.</p></div>', unsafe_allow_html=True)

    section("Why this platform can be differentiated", "A hypothesis to prove through research partnerships and product adoption — not a claim of market exclusivity.")
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="ea-card"><div class="ea-card-label">01 · Paediatric-first context</div><div class="ea-card-value" style="font-size:18px">Disease-specific framing</div><div class="ea-card-note">Keep paediatric patient evidence distinct from adult-derived cell-line and broader experimental evidence.</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="ea-card"><div class="ea-card-label">02 · Evidence lineage</div><div class="ea-card-value" style="font-size:18px">Inspect every connection</div><div class="ea-card-note">Expose source identifiers, releases, transformations, mapping methods, validation and unresolved gaps.</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="ea-card"><div class="ea-card-label">03 · Reproducible research</div><div class="ea-card-value" style="font-size:18px">Preserve the analysis</div><div class="ea-card-note">Build towards versioned research workspaces that can be reviewed, repeated and shared.</div></div>', unsafe_allow_html=True)

    section("What is demonstrated today", "Engineering achievements and data assets, separated from scientific capabilities still being developed.")
    done = [
        ("Graph V3 typed integration", "4,522,696 intergenic observations linked to 3,223 samples; original Graph V2 stores retained and validated."),
        ("Evidence indexing", "SQLite index supports sample-level paginated intergenic lookup."),
        ("Target evidence extract", "ChEMBL drug-target annotations available for record-level inspection; mappings require review."),
        ("Response data", "GDSC2 in-vitro response records available for descriptive exploration."),
    ]
    for title, copy in done:
        st.markdown(f"<div class='ea-step'><div class='ea-step-no'>✓</div><div><div class='ea-step-title'>{title}</div><div class='ea-step-copy'>{copy}</div></div></div>", unsafe_allow_html=True)

    section("What remains to prove", "The next milestones are scientific, not merely visual.")
    future = [
        ("Disease-specific biological rationale", "Validate links among drug targets, paediatric tumour context, pathway activity and functional evidence."),
        ("Predictive response modelling", "Define a response task, prevent cell-line and dataset leakage, benchmark baselines and validate independently."),
        ("Functional evidence for noncoding observations", "Add coordinate-aware regulatory annotation and functional evidence before assigning intergenic observations to genes or pathways."),
        ("External research validation", "Work with domain researchers to assess utility, correctness, reproducibility and workflow impact."),
    ]
    for title, copy in future:
        st.markdown(f"<div class='ea-step'><div class='ea-step-no'>→</div><div><div class='ea-step-title'>{title}</div><div class='ea-step-copy'>{copy}</div></div></div>", unsafe_allow_html=True)

    st.markdown('<div class="ea-callout ea-callout-teal">Investor framing: the current milestone is a validated data-integration and evidence-access foundation. The platform is not yet a clinically validated drug-repurposing engine, and no patient-treatment recommendation is generated.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="ea-footer">EVIDENCE ATLAS · RESEARCH SOFTWARE · Evidence states are source-labelled. Observed data, curated annotations, derived transformations, inferences and predictions are not interchangeable.</div>',
    unsafe_allow_html=True,
)
