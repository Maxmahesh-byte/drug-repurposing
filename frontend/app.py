from __future__ import annotations

import json
import html
import math
import os
import sqlite3
import subprocess
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

from frontend.evidence_network_3d import render_evidence_network_3d
from src.inference.gdsc2_expression import (
    load_expression_model,
    predict_expression_response,
)

DATA = PROJECT_ROOT / "data"
GRAPH_V3_DIR = DATA / "processed" / "graph_v3"
RESEARCH_DB = GRAPH_V3_DIR / "research_evidence.sqlite"
RESEARCH_INDEX_MANIFEST = GRAPH_V3_DIR / "research_evidence_index_manifest.json"
GRAPH_V3_MANIFEST = GRAPH_V3_DIR / "graph_v3_manifest.json"
INTERGENIC_DB = GRAPH_V3_DIR / "intergenic_evidence.sqlite"
MOLECULAR_MODEL_DIR = DATA / "ai" / "molecular_models"
MOLECULAR_MODEL_PATH = MOLECULAR_MODEL_DIR / "gdsc2_expression_ridge.json"
MOLECULAR_MODEL_SUMMARY_PATH = (
    MOLECULAR_MODEL_DIR / "gdsc2_expression_ridge_summary.json"
)
MOLECULAR_PREDICTIONS = {
    "LN(IC50)": MOLECULAR_MODEL_DIR / "gdsc2_expression_ln_ic50_oof_predictions.csv",
    "AUC": MOLECULAR_MODEL_DIR / "gdsc2_expression_auc_oof_predictions.csv",
}

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
/* Thursday demo polish: restrained motion, clearer focus and a stronger research-workbench rhythm. */
.ea-card, .ea-panel, [data-testid="stMetric"] { transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
.ea-card:hover, .ea-panel:hover { transform: translateY(-2px); border-color: #cbd8e3; box-shadow: 0 10px 24px rgba(20,40,62,.07); }
.stButton button:focus-visible, [data-testid="stSelectbox"]:focus-within { outline: 2px solid #13877f; outline-offset: 2px; }
.ea-demo-intro { display:flex; align-items:flex-start; justify-content:space-between; gap:22px; padding:20px 22px; border:1px solid #dfe8ee; border-radius:17px; background:linear-gradient(115deg,#fff 0%,#f4faf9 100%); margin: 8px 0 18px 0; }
.ea-demo-intro h2 { font-size:21px; line-height:1.2; letter-spacing:-.035em; margin:0 0 7px 0; color:#14283e; }
.ea-demo-intro p { font-size:12px; line-height:1.65; max-width:810px; color:#65768b; margin:0; }
.ea-demo-stamp { white-space:nowrap; border:1px solid #cde6e1; border-radius:999px; padding:7px 10px; color:#176d66; background:#e8f5f2; text-transform:uppercase; letter-spacing:.1em; font-size:9px; font-weight:800; }
.ea-demo-card { height:100%; min-height:184px; border:1px solid #e0e7ef; border-radius:17px; padding:19px; background:#fff; box-shadow:0 5px 18px rgba(20,40,62,.035); position:relative; overflow:hidden; }
.ea-demo-card:before { content:""; position:absolute; top:0; left:0; right:0; height:3px; background:#13877f; }
.ea-demo-card.blue:before { background:#5279ad; }
.ea-demo-card.violet:before { background:#7958b5; }
.ea-demo-index { color:#8090a3; font-size:10px; text-transform:uppercase; letter-spacing:.15em; font-weight:800; }
.ea-demo-card h3 { color:#14283e; font-size:17px; letter-spacing:-.025em; margin:12px 0 6px 0; }
.ea-demo-card .value { font-size:31px; line-height:1.05; color:#14283e; letter-spacing:-.05em; font-weight:790; }
.ea-demo-card p { color:#758499; font-size:11px; line-height:1.55; margin:9px 0 0 0; }
.ea-demo-arrow { display:flex; align-items:center; justify-content:center; height:100%; min-height:184px; color:#9babb9; font-size:27px; font-weight:300; }
.ea-mini-label { color:#758499; text-transform:uppercase; letter-spacing:.12em; font-size:9px; font-weight:800; }
.ea-proof-line { padding:10px 0; border-bottom:1px solid #edf1f5; display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
.ea-proof-line:last-child { border-bottom:0; }
.ea-proof-line b { color:#1b344c; font-size:12px; }
.ea-proof-line small { color:#7b899a; font-size:10px; line-height:1.4; display:block; margin-top:3px; }
@media (max-width: 800px) { .ea-demo-intro { flex-direction:column; } .ea-demo-stamp { white-space:normal; } .ea-demo-arrow { min-height:30px; transform:rotate(90deg); } }

/* Evidence Atlas visual system v2: editorial research cockpit, not a generic dashboard. */
.ea-hero { border-radius:24px; padding:30px 32px; min-height:190px; box-shadow:0 16px 38px rgba(20,40,62,.12); }
.ea-hero h1 { font-size:clamp(30px,3.4vw,46px); max-width:920px; }
.ea-hero p { font-size:14px; max-width:800px; }
.ea-section-head { margin-top:28px; margin-bottom:14px; }
.ea-section-head h2 { font-size:20px; letter-spacing:-.035em; }
.ea-section-head p { font-size:12px; }
.ea-card { border-radius:18px; padding:19px 20px; box-shadow:0 5px 20px rgba(20,40,62,.035); }
.ea-card:hover { transform:translateY(-3px); box-shadow:0 13px 28px rgba(20,40,62,.075); }
.ea-card-value { font-size:30px; }
.ea-panel { border-radius:18px; padding:19px 20px; }
.ea-command-hero { display:grid; grid-template-columns:1.08fr .92fr; gap:24px; align-items:stretch; background:linear-gradient(115deg,#ffffff 0%,#f5faf9 58%,#edf4f7 100%); border:1px solid #dfe8ee; border-radius:22px; padding:24px; box-shadow:0 7px 24px rgba(20,40,62,.045); }
.ea-command-copy { display:flex; flex-direction:column; justify-content:center; }
.ea-command-copy .kicker { color:#13877f; font-size:10px; font-weight:850; letter-spacing:.19em; text-transform:uppercase; }
.ea-command-copy h2 { color:#14283e; font-size:clamp(25px,2.4vw,35px); line-height:1.09; letter-spacing:-.05em; margin:10px 0 11px; max-width:650px; }
.ea-command-copy p { color:#66788d; font-size:13px; line-height:1.65; max-width:670px; margin:0; }
.ea-command-pills { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }
.ea-command-pill { display:inline-flex; padding:7px 10px; border:1px solid #dce7e8; border-radius:999px; background:#fff; color:#34566a; font-size:10px; font-weight:750; letter-spacing:.04em; }
.ea-atlas-visual { position:relative; overflow:hidden; min-height:245px; border-radius:18px; padding:18px; background:radial-gradient(circle at 50% 48%,#2c6e72 0%,#1d465c 47%,#14283e 100%); display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr auto 1fr; gap:10px; align-items:center; }
.ea-atlas-visual:before,.ea-atlas-visual:after { content:""; position:absolute; border:1px solid rgba(196,232,226,.18); border-radius:50%; width:240px; height:240px; left:50%; top:50%; transform:translate(-50%,-50%); pointer-events:none; }
.ea-atlas-visual:after { width:330px; height:330px; border-color:rgba(196,232,226,.10); }
.ea-orbit-node { position:relative; z-index:1; background:rgba(255,255,255,.09); border:1px solid rgba(226,244,243,.22); border-radius:12px; padding:10px 11px; backdrop-filter:blur(8px); min-height:62px; }
.ea-orbit-node b { display:block; color:#f4fbfa; font-size:11px; letter-spacing:.01em; }
.ea-orbit-node small { color:#bad1d5; font-size:9px; display:block; margin-top:4px; line-height:1.35; }
.ea-orbit-node .dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:6px; vertical-align:1px; }
.ea-orbit-node.clinical .dot { background:#65c7b8; }
.ea-orbit-node.molecular .dot { background:#9b86df; }
.ea-orbit-node.targets .dot { background:#77a5d9; }
.ea-orbit-node.models .dot { background:#e5b66f; }
.ea-orbit-hub { z-index:2; grid-column:1 / 3; justify-self:center; text-align:center; background:#f4faf9; border:1px solid rgba(255,255,255,.65); border-radius:14px; padding:11px 18px; min-width:180px; box-shadow:0 8px 25px rgba(0,0,0,.18); }
.ea-orbit-hub b { display:block; color:#16384d; font-size:12px; letter-spacing:.08em; }
.ea-orbit-hub small { color:#547181; font-size:9px; display:block; margin-top:3px; }
.ea-metric-ribbon { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:15px 0 22px; }
.ea-ribbon-cell { border:1px solid #e1e8ee; background:#fff; border-radius:14px; padding:13px 15px; min-width:0; }
.ea-ribbon-cell .label { color:#7b8b9d; text-transform:uppercase; letter-spacing:.11em; font-size:9px; font-weight:850; }
.ea-ribbon-cell .number { color:#14283e; font-size:24px; font-weight:790; letter-spacing:-.05em; margin-top:5px; white-space:nowrap; }
.ea-ribbon-cell .note { color:#7b8999; font-size:9px; line-height:1.35; margin-top:3px; }
.ea-question-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:13px; margin:5px 0 10px; }
.ea-question-card { position:relative; overflow:hidden; background:#fff; border:1px solid #e1e8ef; border-radius:17px; padding:18px 19px 17px; min-height:158px; box-shadow:0 5px 18px rgba(20,40,62,.03); }
.ea-question-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:#13877f; }
.ea-question-card.blue:before { background:#5279ad; }
.ea-question-card.violet:before { background:#7958b5; }
.ea-question-card .num { color:#95a3b2; font-size:9px; font-weight:850; letter-spacing:.16em; }
.ea-question-card h3 { color:#172f47; font-size:16px; line-height:1.2; letter-spacing:-.03em; margin:10px 0 7px; }
.ea-question-card p { color:#718196; font-size:11px; line-height:1.55; margin:0; }
.ea-question-card .foot { color:#527080; font-size:9px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin-top:13px; }
.ea-source-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:8px 0 14px; }
.ea-source-tile { background:rgba(255,255,255,.82); border:1px solid #e1e8ef; border-radius:14px; padding:13px 14px; min-width:0; }
.ea-source-tile .top { display:flex; align-items:center; justify-content:space-between; gap:7px; }
.ea-source-tile .top b { color:#263f56; font-size:10px; }
.ea-source-tile .count { color:#14283e; font-size:22px; font-weight:790; letter-spacing:-.045em; margin-top:9px; }
.ea-source-tile .desc { color:#7b899a; font-size:9px; line-height:1.4; margin-top:3px; }
.ea-source-dot { width:7px; height:7px; display:inline-block; border-radius:50%; }
.ea-research-use { background:#f7fafb; border:1px solid #e3eaf0; border-radius:18px; padding:17px 19px; }
.ea-research-use h3 { color:#1a344c; font-size:14px; margin:0 0 10px; }
.ea-use-item { display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-top:1px solid #e8eef2; }
.ea-use-icon { width:24px; height:24px; flex:0 0 24px; border-radius:8px; background:#e5f3f0; color:#147c73; display:flex; justify-content:center; align-items:center; font-size:11px; font-weight:850; }
.ea-use-item b { color:#254057; font-size:11px; display:block; }
.ea-use-item small { color:#7a8999; font-size:10px; line-height:1.4; display:block; margin-top:2px; }
.ea-workbench-head { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; background:linear-gradient(112deg,#ffffff 0%,#f4f9f9 100%); border:1px solid #dfe8ee; border-radius:18px; padding:19px 21px; margin:6px 0 15px; }
.ea-workbench-head h2 { color:#14283e; font-size:21px; letter-spacing:-.04em; margin:5px 0 6px; }
.ea-workbench-head p { color:#6d7f92; font-size:11px; line-height:1.55; margin:0; max-width:760px; }
.ea-workbench-tag { color:#146e67; border:1px solid #d3e9e4; background:#eaf6f3; border-radius:999px; padding:7px 10px; font-size:9px; font-weight:850; letter-spacing:.11em; white-space:nowrap; }
.ea-boundary-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
.ea-boundary-card { border:1px solid #e3eaf0; background:#fff; border-radius:14px; padding:14px 16px; }
.ea-boundary-card h4 { color:#203a52; font-size:11px; margin:0 0 7px; }
.ea-boundary-card p { color:#748498; font-size:10px; line-height:1.5; margin:0; }
@media (max-width: 1050px) { .ea-metric-ribbon { grid-template-columns:repeat(3,1fr); } .ea-source-grid { grid-template-columns:repeat(3,1fr); } }
@media (max-width: 800px) { .ea-command-hero { grid-template-columns:1fr; } .ea-atlas-visual { min-height:230px; } .ea-metric-ribbon { grid-template-columns:repeat(2,1fr); } .ea-question-grid { grid-template-columns:1fr; } .ea-source-grid { grid-template-columns:repeat(2,1fr); } .ea-workbench-head { flex-direction:column; } .ea-boundary-grid { grid-template-columns:1fr; } }


/* Knowledge graph: immersive navy canvas with semantic edge legend. */
.ea-graph-intro { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; padding:21px 23px; border:1px solid #dce7ed; border-radius:18px; background:linear-gradient(115deg,#fff 0%,#f3f8fa 58%,#edf6f5 100%); margin:7px 0 17px; }
.ea-graph-kicker { color:#13877f; font-size:9px; font-weight:850; letter-spacing:.17em; }
.ea-graph-title { color:#14283e; font-size:26px; font-weight:800; letter-spacing:-.05em; margin:6px 0 5px; }
.ea-graph-copy { color:#687b8e; font-size:11px; line-height:1.6; max-width:780px; }
.ea-graph-stamp { color:#176c66; border:1px solid #cde7e1; border-radius:999px; background:#eaf6f3; padding:7px 10px; font-size:9px; font-weight:850; letter-spacing:.1em; white-space:nowrap; }
.ea-graph-canvas { background:#0f2031; border:1px solid #263f55; border-radius:19px; padding:8px; box-shadow:0 18px 38px rgba(13,32,49,.16); overflow:hidden; }
.ea-graph-legend { display:flex; flex-wrap:wrap; gap:9px 18px; padding:11px 14px; margin:8px 0 16px; border:1px solid #e0e8ee; border-radius:13px; background:#fff; }
.ea-graph-legend span { display:flex; align-items:center; gap:7px; color:#53677c; font-size:10px; font-weight:650; }
.ea-graph-legend i { width:17px; height:3px; display:inline-block; border-radius:4px; }
.ea-graph-legend i.map { background:#77a5d9; }
.ea-graph-legend i.record { background:#54beab; }
.ea-graph-legend i.annotation { background:#a08be0; }
.ea-graph-legend i.overlap { height:0; border-top:2px dashed #e1b267; }
.ea-graph-legend i.assay { height:0; border-top:2px dotted #719ddc; }
.ea-graph-detail { border:1px solid #dfe8ed; border-radius:13px; padding:13px 15px; background:#f4faf8; margin:9px 0; }
.ea-graph-detail.blue { background:#f2f6fc; }
.ea-graph-detail b { display:block; color:#19384d; font-size:12px; }
.ea-graph-detail span { display:block; color:#718297; font-size:10px; line-height:1.5; margin-top:4px; }
.ea-graph-boundary { border-left:3px solid #b97824; border-radius:0 12px 12px 0; padding:12px 15px; background:#fff8eb; margin:14px 0; }
.ea-graph-boundary b { color:#78551e; font-size:11px; }
.ea-graph-boundary p { color:#806b49; font-size:10px; line-height:1.6; margin:5px 0 0; }
@media (max-width:850px) { .ea-graph-intro { flex-direction:column; } .ea-graph-stamp { white-space:normal; } }

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
if not clinical.empty and "sample_id" in clinical and "tumor_type" in clinical:
    _label_mask = clinical["tumor_type"].fillna("").astype(str).str.strip().ne("")
    clinical_with_tumor_label_count = int(clinical.loc[_label_mask, "sample_id"].nunique())
    clinical_without_tumor_label_count = max(0, clinical_count - clinical_with_tumor_label_count)
else:
    clinical_with_tumor_label_count = 0
    clinical_without_tumor_label_count = 0
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
            "Cohort intelligence",
            "Molecular evidence workbench",
            "Molecular response model",
            "Evidence dossier",
            "Knowledge graph",
            "Sample explorer",
            "Data integrity",
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
    st.markdown("<div class='ea-sidebar-note'>An experimental single-agent molecular-response benchmark is available in its own workspace. No clinical treatment recommendations are generated.</div>", unsafe_allow_html=True)

# Common hero banner.
page_heroes = {
    "Command centre": ("RESEARCH OVERVIEW", "Evidence, connected with context.", "A live view of the platform's current research assets, integration status and evidence coverage."),
    "Cohort intelligence": ("COHORT INTELLIGENCE STUDIO", "Understand the population behind each label.", "Explore the reported tumour landscape, label coverage and gene-associated observations while preserving source definitions."),
    "Molecular evidence workbench": ("MOLECULAR EVIDENCE WORKBENCH", "Build an evidence dossier, one source at a time.", "Inspect ChEMBL target annotations, OpenPedCan molecular observations and separate GDSC2 model assays."),
    "Molecular response model": ("EXPERIMENTAL RESPONSE MODELLING", "Inspect a molecular model against held-out cell lines.", "Review its measured-data benchmark, compare it with response-only baselines, and explore cell-line expression estimates. This is not a patient-treatment predictor."),
    "Evidence dossier": ("INVESTIGATION WORKSPACE", "Interrogate the evidence behind a research question.", "Separate target records, tumour molecular context and in-vitro response observations. No composite efficacy score is generated."),
    "Knowledge graph": ("EVIDENCE RELATIONSHIP GRAPH", "See the research landscape as a connected system.", "Navigate source-backed relationships across compounds, target annotations, gene-associated observations and experimental models. Every edge keeps its evidence meaning."),
    "Sample explorer": ("SAMPLE-LEVEL EVIDENCE", "Move from cohort context to individual observations.", "Inspect intergenic records with source-row provenance and bounded pagination."),
    "Data integrity": ("DATA INTEGRITY", "Every observation should have a history.", "Inspect source releases, checksums, coordinate conventions and validation boundaries."),
    "Investor briefing": ("PLATFORM THESIS", "A paediatric oncology evidence intelligence layer.", "A clear view of what is built, what is differentiated and what remains to be demonstrated."),
}
hero_kicker, hero_title, hero_copy = page_heroes[page]
if page != "Command centre":
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
    label_coverage = (
        clinical_with_tumor_label_count / clinical_count
        if clinical_count else 0.0
    )

    # Editorial command hero + architectural evidence constellation.
    st.markdown(
        '<div class="ea-command-hero">'
        '<div class="ea-command-copy">'
        '<div class="kicker">THE PAEDIATRIC ONCOLOGY RESEARCH WORKSPACE</div>'
        '<h2>See the evidence behind the research question.</h2>'
        '<p>Evidence Atlas brings clinical context, molecular observations, drug annotations and experimental model data into one inspectable workspace. Each evidence lane keeps its source, meaning and limitations visible.</p>'
        '<div class="ea-command-pills">'
        '<span class="ea-command-pill">Source traceability</span>'
        '<span class="ea-command-pill">Paediatric-first context</span>'
        '<span class="ea-command-pill">Reproducible exploration</span>'
        '</div></div>'
        '<div class="ea-atlas-visual">'
        '<div class="ea-orbit-node clinical"><b><span class="dot"></span>Clinical context</b><small>OpenPedCan · cohort and diagnosis labels</small></div>'
        '<div class="ea-orbit-node molecular"><b><span class="dot"></span>Molecular observations</b><small>Variants · sample-level evidence</small></div>'
        '<div class="ea-orbit-hub"><b>◈ EVIDENCE ATLAS</b><small>Context · provenance · uncertainty</small></div>'
        '<div class="ea-orbit-node targets"><b><span class="dot"></span>Drug annotations</b><small>ChEMBL · molecule and target records</small></div>'
        '<div class="ea-orbit-node models"><b><span class="dot"></span>Experimental models</b><small>GDSC2 · in-vitro response measurements</small></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Source populations are deliberately separated; no sum across evidence types.
    st.markdown(
        '<div class="ea-metric-ribbon">'
        f'<div class="ea-ribbon-cell"><div class="label">Clinical sample IDs</div><div class="number">{clinical_count:,}</div><div class="note">OpenPedCan metadata</div></div>'
        f'<div class="ea-ribbon-cell"><div class="label">Gene-associated mutations</div><div class="number">{int(index_manifest.get("counts", {}).get("gene_associated_variant_rows", 0) or 0):,}</div><div class="note">Indexed observations</div></div>'
        f'<div class="ea-ribbon-cell"><div class="label">Intergenic observations</div><div class="number">{intergenic_count:,}</div><div class="note">Graph V3 · {intergenic_sample_count:,} matched sample barcodes</div></div>'
        f'<div class="ea-ribbon-cell"><div class="label">Drug-target rows</div><div class="number">{chembl_count:,}</div><div class="note">ChEMBL target evidence extract</div></div>'
        f'<div class="ea-ribbon-cell"><div class="label">Model response rows</div><div class="number">{gdsc_count:,}</div><div class="note">GDSC2 in-vitro measurements</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Three questions the workspace helps researchers explore", "The prototype organises evidence around practical research tasks — without turning incomplete evidence into a prediction.")
    st.markdown(
        '<div class="ea-question-grid">'
        '<div class="ea-question-card"><div class="num">01 · COMPOUND CONTEXT</div><h3>What is recorded about the target?</h3><p>Inspect compound identifiers, source-reported target annotations, mechanisms and gene-symbol mappings.</p><div class="foot">ChEMBL · molecule → target records</div></div>'
        '<div class="ea-question-card blue"><div class="num">02 · DISEASE CONTEXT</div><h3>What is observed in this disease?</h3><p>Explore tumour-associated molecular observations, sample denominators and exact gene-symbol overlaps.</p><div class="foot">OpenPedCan · cohort → observations</div></div>'
        '<div class="ea-question-card violet"><div class="num">03 · MODEL CONTEXT</div><h3>What do experimental models show?</h3><p>Inspect in-vitro response distributions and model labels while keeping them separate from patient evidence.</p><div class="foot">GDSC2 · cell line → assay records</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Evidence source landscape", "A compact view of the current data assets. Counts belong to different source populations and should not be added together.")
    st.markdown(
        '<div class="ea-source-grid">'
        f'<div class="ea-source-tile"><div class="top"><b>OpenPedCan clinical</b><span class="ea-source-dot" style="background:#5279ad"></span></div><div class="count">{clinical_count:,}</div><div class="desc">Unique sample IDs · {clinical_with_tumor_label_count:,} have a populated tumour label</div></div>'
        f'<div class="ea-source-tile"><div class="top"><b>OpenPedCan variants</b><span class="ea-source-dot" style="background:#7958b5"></span></div><div class="count">{mutation_count:,}</div><div class="desc">Source mutation observations scanned / matched</div></div>'
        f'<div class="ea-source-tile"><div class="top"><b>Graph V3</b><span class="ea-source-dot" style="background:#13877f"></span></div><div class="count">{intergenic_count:,}</div><div class="desc">Intergenic observations with sample-linked provenance</div></div>'
        f'<div class="ea-source-tile"><div class="top"><b>ChEMBL extract</b><span class="ea-source-dot" style="background:#5279ad"></span></div><div class="count">{chembl_count:,}</div><div class="desc">Drug–molecule–target annotation rows</div></div>'
        f'<div class="ea-source-tile"><div class="top"><b>GDSC2</b><span class="ea-source-dot" style="background:#b97824"></span></div><div class="count">{gdsc_count:,}</div><div class="desc">Cell-line in-vitro response measurements</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not research_ready:
        st.markdown(
            '<div class="ea-callout">The compact research index has not been built yet. The overview remains available, but disease-specific gene overlap and response distributions require the validated index.</div>',
            unsafe_allow_html=True,
        )
        st.code("./.venv/bin/python scripts/build_research_evidence_index.py\n./.venv/bin/python scripts/validate_research_evidence_index.py", language="bash")

    section("Cohort intelligence", "Start with the disease labels actually present in the clinical metadata. Missing labels remain visible.")
    if not clinical.empty and "tumor_type" in clinical and "sample_id" in clinical:
        cohort_counts = (
            clinical.dropna(subset=["tumor_type"])
            .assign(tumor_type=lambda d: d["tumor_type"].astype(str).str.strip())
            .query("tumor_type != ''")
            .groupby("tumor_type")["sample_id"].nunique()
            .sort_values(ascending=False).head(18).reset_index(name="biospecimens")
        )
        chart_col, coverage_col = st.columns([1.55, .85], gap="large")
        with chart_col:
            st.markdown('<div class="ea-panel"><div class="ea-panel-title">Tumour label landscape</div><div class="ea-panel-sub">Unique sample IDs by reported OpenPedCan tumour label · top 18 labels</div>', unsafe_allow_html=True)
            if PLOTLY_AVAILABLE and not cohort_counts.empty:
                fig = px.treemap(
                    cohort_counts, path=["tumor_type"], values="biospecimens",
                    color="biospecimens",
                    color_continuous_scale=["#eaf4f2", "#9fcfc6", "#13877f"],
                    labels={"tumor_type": "Tumour label", "biospecimens": "Unique sample IDs"},
                )
                fig.update_traces(textinfo="label+value", marker=dict(line=dict(color="#ffffff", width=2)), tiling=dict(pad=3))
                fig.update_layout(coloraxis_showscale=False)
                style_figure(fig, 365)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            elif not cohort_counts.empty:
                st.bar_chart(cohort_counts.set_index("tumor_type"))
            else:
                st.info("No populated tumour labels are available in the clinical metadata.")
            st.markdown('</div>', unsafe_allow_html=True)
        with coverage_col:
            st.markdown('<div class="ea-panel"><div class="ea-panel-title">Clinical label coverage</div><div class="ea-panel-sub">Coverage is a data-quality measure, not disease completeness.</div>', unsafe_allow_html=True)
            if PLOTLY_AVAILABLE and clinical_count:
                fig = go.Figure(go.Pie(
                    labels=["Tumour label present", "Tumour label missing"],
                    values=[clinical_with_tumor_label_count, clinical_without_tumor_label_count],
                    hole=.78, sort=False,
                    marker=dict(colors=["#13877f", "#e7edf2"], line=dict(color="#ffffff", width=2)),
                    textinfo="none", hovertemplate="%{label}<br>%{value:,} sample IDs<extra></extra>",
                ))
                fig.add_annotation(text=f"{label_coverage:.1%}", x=.5, y=.53, showarrow=False, font=dict(size=27, color="#14283e", family="Inter, sans-serif"))
                fig.add_annotation(text="labelled", x=.5, y=.40, showarrow=False, font=dict(size=10, color="#718196"))
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-.08, x=.02, font=dict(size=9)), margin=dict(l=5,r=5,t=15,b=25), height=255, paper_bgcolor="rgba(255,255,255,0)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.metric("Sample IDs with tumour label", f"{clinical_with_tumor_label_count:,}")
            st.markdown(
                f'<div class="ea-proof-line"><div><b>Label present</b><small>OpenPedCan clinical metadata</small></div><strong>{clinical_with_tumor_label_count:,}</strong></div>'
                f'<div class="ea-proof-line"><div><b>Label missing</b><small>Not silently inferred</small></div><strong>{clinical_without_tumor_label_count:,}</strong></div>'
                f'<div class="ea-proof-line"><div><b>Total unique sample IDs</b><small>Clinical metadata denominator</small></div><strong>{clinical_count:,}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("OpenPedCan clinical metadata was not found at the expected path.")

    section("How this helps a researcher", "The value is not another count of records — it is a more inspectable route through fragmented evidence.")
    use_left, use_right = st.columns([1.1, 1], gap="large")
    with use_left:
        st.markdown(
            '<div class="ea-research-use"><h3>From a broad question to inspectable evidence</h3>'
            '<div class="ea-use-item"><div class="ea-use-icon">01</div><div><b>Orient quickly</b><small>See which evidence domains are present before beginning a deeper investigation.</small></div></div>'
            '<div class="ea-use-item"><div class="ea-use-icon">02</div><div><b>Follow identifiers and mappings</b><small>Move between source drug records, target annotations, tumour labels and sample-linked observations.</small></div></div>'
            '<div class="ea-use-item"><div class="ea-use-icon">03</div><div><b>Find the gaps</b><small>Expose missing labels, unresolved mappings and absent evidence rather than hiding them behind a score.</small></div></div>'
            '<div class="ea-use-item"><div class="ea-use-icon">04</div><div><b>Reproduce the inspection</b><small>Return to the underlying source rows, release context and indexed records.</small></div></div></div>',
            unsafe_allow_html=True,
        )
    with use_right:
        st.markdown(
            '<div class="ea-panel"><div class="ea-panel-title">Evidence semantics</div><div class="ea-panel-sub">Colour is a cue; the label carries the meaning.</div><hr class="ea-rule">'
            f'{status_chip("OBSERVED", "observed")} <span style="font-size:11px;color:#65768b;margin-left:8px">Reported by a source dataset</span><br><br>'
            f'{status_chip("CURATED", "curated")} <span style="font-size:11px;color:#65768b;margin-left:8px">Mapped or annotated in a source release</span><br><br>'
            f'{status_chip("DERIVED", "derived")} <span style="font-size:11px;color:#65768b;margin-left:8px">Produced by a documented transformation</span><br><br>'
            f'{status_chip("INFERRED", "inferred")} <span style="font-size:11px;color:#65768b;margin-left:8px">A proposed relationship, not an observation</span><br><br>'
            f'{status_chip("CONFLICT", "conflict")} <span style="font-size:11px;color:#65768b;margin-left:8px">Source labels or evidence disagree</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="ea-callout ea-callout-teal">The prototype is designed to make research navigation more transparent. It does not establish biological causality, target dependency, treatment efficacy or clinical benefit.</div>',
        unsafe_allow_html=True,
    )


elif page == "Cohort intelligence":
    if clinical.empty or "tumor_type" not in clinical or "sample_id" not in clinical:
        st.warning("OpenPedCan clinical metadata is not available at the expected path.")
    else:
        cohort_all = (
            clinical.assign(tumor_type=lambda d: d["tumor_type"].fillna("").astype(str).str.strip())
            .assign(sample_id=lambda d: d["sample_id"].fillna("").astype(str).str.strip())
        )
        labelled = cohort_all[cohort_all["tumor_type"].ne("") & cohort_all["sample_id"].ne("")]
        cohort_counts_all = (
            labelled.groupby("tumor_type")["sample_id"].nunique()
            .sort_values(ascending=False).rename("biospecimens").reset_index()
        )
        label_names = cohort_counts_all["tumor_type"].astype(str).tolist()
        if not label_names:
            st.info("No populated tumour labels are present in the clinical metadata.")
        else:
            selected_label = st.selectbox(
                "Explore an OpenPedCan tumour label",
                label_names,
                index=0,
                key="ea_cohort_label",
            )
            selected_label_html = html.escape(str(selected_label))
            selected_count = int(
                cohort_counts_all.loc[
                    cohort_counts_all["tumor_type"] == selected_label, "biospecimens"
                ].iloc[0]
            )
            labelled_ids = int(labelled["sample_id"].nunique())
            missing_ids = max(0, clinical_count - labelled_ids)
            cohort_unique_labels = int(cohort_counts_all["tumor_type"].nunique())

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Unique clinical sample IDs", f"{clinical_count:,}")
            with m2: st.metric("IDs with tumour label", f"{labelled_ids:,}")
            with m3: st.metric("Distinct reported labels", f"{cohort_unique_labels:,}")
            with m4: st.metric("Selected label sample IDs", f"{selected_count:,}")

            st.markdown(
                '<div class="ea-callout ea-callout-blue">These are source-reported OpenPedCan tumour labels, not an approved disease ontology or a harmonised cross-dataset crosswalk. Labels and denominators remain tied to this source release.</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.25, 1], gap="large")
            with left:
                section("Reported tumour landscape", "Top 24 labels by unique sample IDs in the clinical metadata.")
                plot_data = cohort_counts_all.head(24).sort_values("biospecimens")
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        plot_data,
                        x="biospecimens", y="tumor_type", orientation="h",
                        color="biospecimens",
                        color_continuous_scale=["#dceeea", "#13877f"],
                        labels={"biospecimens": "Unique sample IDs", "tumor_type": "Reported tumour label"},
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    style_figure(fig, 540)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.bar_chart(plot_data.set_index("tumor_type"))

            with right:
                section("Selected cohort snapshot", "A source-specific cohort slice, not a statement of biological equivalence.")
                st.markdown(
                    f'<div class="ea-panel"><div class="ea-card-label">SELECTED OPENPEDCAN LABEL</div>'
                    f'<div class="ea-card-value" style="font-size:24px">{selected_label_html}</div>'
                    f'<div class="ea-card-note">{selected_count:,} unique sample IDs in the clinical metadata.</div>'
                    f'<hr class="ea-rule"><div class="ea-proof-line"><div><b>Source</b><small>OpenPedCan clinical metadata</small></div><strong>Clinical</strong></div>'
                    f'<div class="ea-proof-line"><div><b>Label coverage</b><small>Missing labels retained separately</small></div><strong>{labelled_ids:,} / {clinical_count:,}</strong></div>'
                    f'<div class="ea-proof-line"><div><b>Unlabelled sample IDs</b><small>Not assigned by inference</small></div><strong>{missing_ids:,}</strong></div></div>',
                    unsafe_allow_html=True,
                )
                if research_ready:
                    gene_summary = get_disease_gene_summary(selected_label, 18)
                    section("Gene-associated observations", "Indexed observations for this exact tumour label. These are not pathogenicity or dependency scores.")
                    if gene_summary.empty:
                        st.info("No gene-associated mutation summary rows were returned for this exact label.")
                    elif PLOTLY_AVAILABLE:
                        fig = px.bar(
                            gene_summary.sort_values("observations"),
                            x="observations", y="gene_symbol", orientation="h",
                            color_discrete_sequence=[PALETTE["violet"]],
                            labels={"observations": "Gene-associated observations", "gene_symbol": "Gene symbol"},
                        )
                        style_figure(fig, 430)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.dataframe(gene_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("Build and validate the research evidence index to inspect gene-associated observations.")

            section("How to interpret this view", "A useful cohort starting point, not a biological validation result.")
            st.markdown(
                '<div class="ea-boundary-grid">'
                '<div class="ea-boundary-card"><h4>WHAT IS SUPPORTED</h4><p>Reported tumour-label counts, unique sample-ID denominators, missing-label coverage and indexed gene-associated observation counts.</p></div>'
                '<div class="ea-boundary-card"><h4>WHAT IS NOT ESTABLISHED</h4><p>Cross-dataset disease equivalence, patient-level prevalence, variant pathogenicity, gene dependency or therapeutic response.</p></div>'
                '</div>',
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Page 3: Evidence dossier
# -----------------------------------------------------------------------------
elif page == "Evidence dossier":
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
# Page 5: Knowledge graph — source-backed links with explicit edge semantics.
# -----------------------------------------------------------------------------
elif page == "Knowledge graph":
    # Give the graph workspace more horizontal room than the standard dashboard pages.
    st.markdown(
        """<style>
        .block-container { max-width: 100vw !important; padding-left: .8rem !important;
            padding-right: .8rem !important; padding-top: .55rem !important;
            padding-bottom: .8rem !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    section(
        "Compound-to-tumour evidence network",
        "Follow the source records from a GDSC compound mapping through ChEMBL target annotations and exact gene-symbol overlap into an OpenPedCan tumour-labelled summary. GDSC2 model assays remain separate.",
    )
    st.markdown(
        '<div class="ea-callout ea-callout-blue">Dashed amber links indicate exact normalized gene-symbol overlap in the selected tumour summary. They do not imply a causal or therapeutic relationship. Protein-complex gene symbols may represent constituent subunits of one target complex.</div>',
        unsafe_allow_html=True,
    )

    if not research_ready:
        st.warning("Build the research evidence index before opening the 3D Knowledge Graph.")
    elif clinical.empty or "tumor_type" not in clinical:
        st.warning("OpenPedCan clinical tumour labels are unavailable in the current local source file.")
    else:
        drugs = get_drugs()
        disease_options = sorted(
            clinical["tumor_type"].dropna().astype(str).str.strip().unique().tolist(),
            key=str.casefold,
        )
        disease_options = [value for value in disease_options if value]

        if drugs.empty:
            st.warning("No GDSC2 drug records are available in the research index.")
        elif not disease_options:
            st.warning("No populated OpenPedCan tumour labels were found.")
        else:
            drug_options = drugs.drop_duplicates(subset=["drug_id", "drug_name"]).copy()
            drug_options["_label"] = drug_options.apply(
                lambda r: f"{str(r['drug_name']).strip()} · GDSC ID {str(r['drug_id']).strip()}",
                axis=1,
            )
            drug_labels = drug_options["_label"].tolist()
            default_drug_index = 0
            bortezomib_matches = [
                i for i, value in enumerate(drug_labels)
                if "bortezomib" in value.casefold()
            ]
            if bortezomib_matches:
                default_drug_index = bortezomib_matches[0]

            default_disease_index = (
                disease_options.index("Neuroblastoma")
                if "Neuroblastoma" in disease_options else 0
            )

            filter_cols = st.columns([2.2, 1.8, 1.0])
            with filter_cols[0]:
                chosen_drug_label = st.selectbox(
                    "Compound / GDSC drug record",
                    drug_labels,
                    index=default_drug_index,
                    key="ea_kg3d_drug_v3",
                )
            with filter_cols[1]:
                chosen_disease = st.selectbox(
                    "OpenPedCan tumour label",
                    disease_options,
                    index=default_disease_index,
                    key="ea_kg3d_disease_v3",
                )
            with filter_cols[2]:
                visible_gene_limit = st.selectbox(
                    "Visible genes",
                    [8, 12, 16, 20],
                    index=1,
                    key="ea_kg3d_gene_limit_v3",
                )

            selected_drug = drug_options.loc[
                drug_options["_label"] == chosen_drug_label
            ].iloc[0]
            selected_drug_id = str(selected_drug["drug_id"]).strip()
            selected_drug_name = str(selected_drug["drug_name"]).strip()

            # Source A: GDSC drug record to ChEMBL molecule and target annotations.
            target_records = get_chembl_targets(
                selected_drug_id,
                selected_drug_name,
            )
            target_rows = target_records.to_dict("records") if not target_records.empty else []

            def _kg_clean(value):
                if value is None or pd.isna(value):
                    return ""
                return str(value).strip()

            def _kg_symbols(value):
                raw = _kg_clean(value).replace(";", ",").replace("|", ",")
                return sorted({
                    part.strip().upper()
                    for part in raw.split(",")
                    if part.strip() and part.strip().upper() not in {"UNKNOWN", "NAN", "NONE"}
                })

            molecule_ids = sorted({
                _kg_clean(row.get("molecule_chembl_id"))
                for row in target_rows
                if _kg_clean(row.get("molecule_chembl_id"))
            })
            target_groups = {}
            target_gene_map = {}
            molecule_target_pairs = set()
            all_target_genes = set()
            for row in target_rows:
                molecule_id = _kg_clean(row.get("molecule_chembl_id"))
                target_id = _kg_clean(row.get("target_chembl_id"))
                target_name = _kg_clean(row.get("target_name")) or target_id or "Unresolved target"
                target_type = _kg_clean(row.get("target_type")) or "Target type not supplied"
                target_key = target_id or target_name
                if target_key not in target_groups:
                    target_groups[target_key] = {
                        "id": target_id or target_name,
                        "name": target_name,
                        "type": target_type,
                        "molecules": set(),
                        "records": 0,
                    }
                    target_gene_map[target_key] = set()
                target_groups[target_key]["records"] += 1
                if molecule_id:
                    target_groups[target_key]["molecules"].add(molecule_id)
                    molecule_target_pairs.add((molecule_id, target_key))
                symbols = _kg_symbols(row.get("gene_symbol"))
                target_gene_map[target_key].update(symbols)
                all_target_genes.update(symbols)

            # Source B: the selected OpenPedCan tumour label and mutation summary.
            clinical_rows = clinical[
                clinical["tumor_type"].astype(str).str.strip().str.casefold()
                == chosen_disease.casefold()
            ]
            labelled_biospecimens = (
                int(clinical_rows["sample_id"].nunique())
                if "sample_id" in clinical_rows else 0
            )
            tumour_summary_meta = query_frame(
                """SELECT COUNT(DISTINCT gene_symbol) AS gene_count,
                          COALESCE(SUM(observation_count), 0) AS observation_count
                   FROM mutation_gene_summary
                   WHERE LOWER(TRIM(tumor_type)) = LOWER(TRIM(?))""",
                (chosen_disease,),
            )
            indexed_gene_count = (
                int(tumour_summary_meta.iloc[0]["gene_count"] or 0)
                if not tumour_summary_meta.empty else 0
            )
            tumour_observation_count = (
                int(tumour_summary_meta.iloc[0]["observation_count"] or 0)
                if not tumour_summary_meta.empty else 0
            )
            overlap = (
                get_target_gene_overlap(chosen_disease, tuple(sorted(all_target_genes)))
                if all_target_genes else pd.DataFrame(columns=["gene_symbol", "observations"])
            )
            overlap_lookup = {}
            if not overlap.empty:
                for row in overlap.to_dict("records"):
                    symbol = _kg_clean(row.get("gene_symbol")).upper()
                    try:
                        overlap_lookup[symbol] = int(float(row.get("observations") or 0))
                    except (TypeError, ValueError):
                        overlap_lookup[symbol] = 0
            overlap_symbols = set(overlap_lookup)

            # Prioritise exact overlaps, then display additional annotated symbols.
            overlap_order = sorted(overlap_symbols, key=lambda s: overlap_lookup.get(s, 0), reverse=True)
            remaining_order = sorted(all_target_genes - overlap_symbols)
            display_gene_symbols = (overlap_order + remaining_order)[:int(visible_gene_limit)]
            additional_gene_count = max(0, len(all_target_genes) - len(display_gene_symbols))

            # Source C: GDSC2 response measurements. This branch stays separate
            # from OpenPedCan; no cancer-label equivalence is assumed here.
            response_count = get_drug_response_count(
                selected_drug_id,
                selected_drug_name,
                "All reported cancer types",
            )
            response_types = query_frame(
                """SELECT COALESCE(NULLIF(TRIM(reported_cancer_type), ''), 'Unlabelled') AS model_label,
                          COUNT(*) AS response_records,
                          COUNT(DISTINCT cell_line) AS cell_lines
                   FROM gdsc_response
                   WHERE drug_id = ?
                      OR LOWER(TRIM(drug_name)) = LOWER(TRIM(?))
                   GROUP BY COALESCE(NULLIF(TRIM(reported_cancer_type), ''), 'Unlabelled')
                   ORDER BY response_records DESC
                   LIMIT 5""",
                (selected_drug_id, selected_drug_name),
            )

            # Construct a spatially organised 3D graph. XYZ positions describe
            # display layout only; edges carry the source semantics below.
            nodes = []
            links = []

            def _add_node(
                node_id, label, display_label, kind, color, x, y, z,
                detail, source, radius=12, label_size=9, bold=False, label_side="right",
            ):
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "displayLabel": display_label,
                    "kind": kind,
                    "color": color,
                    "x": float(x), "y": float(y), "z": float(z),
                    "detail": detail,
                    "source": source,
                    "radius": radius,
                    "labelSize": label_size,
                    "bold": bold,
                    "labelSide": label_side,
                })

            def _add_link(source_id, target_id, label, detail, kind, color, dashed=False, curve_side=1):
                links.append({
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "detail": detail,
                    "kind": kind,
                    "color": color,
                    "dashed": bool(dashed),
                    "curveSide": curve_side,
                })

            _add_node(
                "drug:selected",
                selected_drug_name,
                f"{selected_drug_name}\nGDSC ID {selected_drug_id}",
                "GDSC compound record",
                "#79afe1", -610, 55, 10,
                f"GDSC drug record {selected_drug_id} ({selected_drug_name}). This record is the starting point for the ChEMBL mapping extract and the separate GDSC2 response query.",
                "research_evidence.sqlite · gdsc_response and chembl_target_evidence",
                radius=26, label_size=11, bold=True, label_side="bottom",
            )

            # Molecule nodes and the GDSC-to-ChEMBL mapping edge.
            shown_molecules = molecule_ids[:5]
            if shown_molecules:
                molecule_y = [0] if len(shown_molecules) == 1 else [140 - i * (280 / (len(shown_molecules) - 1)) for i in range(len(shown_molecules))]
                for index, molecule_id in enumerate(shown_molecules):
                    molecule_rows = [r for r in target_rows if _kg_clean(r.get("molecule_chembl_id")) == molecule_id]
                    methods = sorted({
                        _kg_clean(r.get("drug_match_method"))
                        for r in molecule_rows if _kg_clean(r.get("drug_match_method"))
                    })
                    method_label = ", ".join(methods) if methods else "Mapping method not supplied"
                    node_id = f"molecule:{molecule_id}"
                    _add_node(
                        node_id, molecule_id, f"{molecule_id}\nChEMBL MOLECULE",
                        "ChEMBL molecule", "#79afe1", -390, molecule_y[index],
                        (index - (len(shown_molecules)-1)/2) * 54,
                        f"ChEMBL molecule identifier {molecule_id}. It is linked to GDSC drug record {selected_drug_id} ({selected_drug_name}) in the local target-evidence extract. Mapping method: {method_label}.",
                        "research_evidence.sqlite · chembl_target_evidence · molecule_chembl_id / drug_match_method",
                        radius=16, label_size=9, label_side="bottom",
                    )
                    _add_link(
                        "drug:selected", node_id,
                        "GDSC drug ↔ ChEMBL molecule mapping",
                        f"The local extract links GDSC drug {selected_drug_id} ({selected_drug_name}) to molecule {molecule_id}. Mapping method recorded: {method_label}. This is a source mapping, not a tumour-specific efficacy relationship.",
                        "compound_mapping", "#79afe1", False, 1 if index % 2 == 0 else -1,
                    )
                if len(molecule_ids) > len(shown_molecules):
                    extra = len(molecule_ids) - len(shown_molecules)
                    _add_node(
                        "molecule:additional", f"{extra} additional molecules",
                        f"+{extra} MORE\nMOLECULES", "Collapsed molecule records",
                        "#6f96b7", -390, -205, -90,
                        f"{extra} additional distinct ChEMBL molecule identifiers are present in the mapping extract but collapsed for graph readability.",
                        "research_evidence.sqlite · chembl_target_evidence",
                        radius=10, label_size=8,
                    )
            else:
                _add_node(
                    "gap:chembl", "No ChEMBL molecule mapping rows",
                    "NO CHemBL\nMOLECULE MAPPING", "Missing source mapping",
                    "#d28d8d", -350, 35, 0,
                    f"No molecule_chembl_id rows were found for GDSC drug {selected_drug_id} ({selected_drug_name}) in the local ChEMBL target-evidence extract.",
                    "research_evidence.sqlite · chembl_target_evidence",
                    radius=17, label_size=9,
                )
                _add_link(
                    "drug:selected", "gap:chembl", "No mapped molecule row",
                    "The selected GDSC drug record returned no linked ChEMBL molecule row in this extract. This is a missing mapping in the current index, not proof that no biological target exists.",
                    "gap", "#d28d8d", True,
                )

            # Target entities are grouped by ChEMBL target identifier. Multiple
            # gene symbols can be components of one target complex.
            target_keys = sorted(target_groups, key=lambda k: (target_groups[k]["name"].casefold(), k))
            shown_target_keys = target_keys[:7]
            if shown_target_keys:
                target_y = [0] if len(shown_target_keys) == 1 else [170 - i * (340 / (len(shown_target_keys)-1)) for i in range(len(shown_target_keys))]
                for index, target_key in enumerate(shown_target_keys):
                    target = target_groups[target_key]
                    target_node_id = f"target:{target_key}"
                    name_short = target["name"] if len(target["name"]) <= 24 else target["name"][:22] + "…"
                    _add_node(
                        target_node_id,
                        target["name"],
                        f"{name_short}\n{target['type']}",
                        "ChEMBL target entity",
                        "#b49aef", -130, target_y[index], (index % 3 - 1) * 90,
                        f"Target {target['id']} · {target['name']} · type {target['type']}. The local extract contains {target['records']:,} associated rows. Where target_type is PROTEIN COMPLEX, listed gene symbols can be constituent subunits of one complex rather than independent targets.",
                        "research_evidence.sqlite · chembl_target_evidence · target_chembl_id / target_type / target_name",
                        radius=18, label_size=8, label_side="top",
                    )
                    for molecule_id in sorted(target["molecules"]):
                        if molecule_id in shown_molecules:
                            molecule_source_node = f"molecule:{molecule_id}"
                        elif len(molecule_ids) > len(shown_molecules):
                            molecule_source_node = "molecule:additional"
                        else:
                            molecule_source_node = ""
                        if molecule_source_node:
                            _add_link(
                                molecule_source_node, target_node_id,
                                "ChEMBL target annotation" if molecule_id in shown_molecules else "Additional molecule records → target",
                                f"The target evidence rows associate molecule {molecule_id} with target {target['id']} ({target['name']}); target type: {target['type']}.",
                                "target_annotation", "#b49aef", molecule_id not in shown_molecules, 1 if index % 2 == 0 else -1,
                            )

                if len(target_keys) > len(shown_target_keys):
                    extra_targets = len(target_keys) - len(shown_target_keys)
                    _add_node(
                        "target:additional", f"{extra_targets} additional target entities",
                        f"+{extra_targets} MORE\nTARGETS", "Collapsed target entities",
                        "#8674aa", -120, -245, -90,
                        f"{extra_targets} additional distinct target entities are in the source mapping but collapsed to keep the graph readable.",
                        "research_evidence.sqlite · chembl_target_evidence",
                        radius=11, label_size=8,
                    )

            # Gene symbols are displayed once, while retaining their links to
            # each target entity. Exact overlap is a dashed descriptive edge.
            gene_y = [0] if len(display_gene_symbols) == 1 else [230 - i * (460 / max(1, len(display_gene_symbols)-1)) for i in range(len(display_gene_symbols))]
            target_key_for_gene = {}
            for target_key, symbols in target_gene_map.items():
                for symbol in symbols:
                    target_key_for_gene.setdefault(symbol, set()).add(target_key)

            if display_gene_symbols:
                for index, symbol in enumerate(display_gene_symbols):
                    observations = overlap_lookup.get(symbol, 0)
                    is_overlap = symbol in overlap_symbols
                    gene_node_id = f"gene:{symbol}"
                    _add_node(
                        gene_node_id,
                        symbol,
                        f"{symbol}\n{str(observations) + ' OBS' if is_overlap else 'TARGET SYMBOL'}",
                        "Exact overlapping gene symbol" if is_overlap else "Target-associated gene symbol",
                        "#58c6b5" if is_overlap else "#9a86c9",
                        115, gene_y[index], ((index % 4) - 1.5) * 82,
                        (
                            f"Exact normalized gene-symbol overlap: {symbol} occurs in the selected tumour-label mutation_gene_summary with {observations:,} aggregated gene-associated observations. This is a descriptive record overlap and does not establish gene dependency, target engagement or causal relevance."
                            if is_overlap else
                            f"{symbol} appears in the selected ChEMBL target annotation records, but an exact normalized symbol match was not returned by the OpenPedCan mutation_gene_summary query for tumour_type={chosen_disease!r}. This does not prove biological absence."
                        ),
                        "ChEMBL target evidence · gene_symbol; comparison: research_evidence.sqlite · mutation_gene_summary",
                        radius=13 if is_overlap else 11, label_size=9, label_side=("left" if index % 2 else "right"),
                    )
                    for target_key in sorted(target_key_for_gene.get(symbol, set())):
                        if target_key in shown_target_keys:
                            target_source_node = f"target:{target_key}"
                        elif len(target_keys) > len(shown_target_keys):
                            target_source_node = "target:additional"
                        else:
                            target_source_node = ""
                        if target_source_node:
                            _add_link(
                                target_source_node, gene_node_id,
                                "Target record → gene-symbol annotation" if target_key in shown_target_keys else "Collapsed targets → gene-symbol annotation",
                                f"Gene symbol {symbol} is attached to target annotation record(s). If the target is a protein complex, this symbol may identify a component/subunit.",
                                "gene_annotation", "#58c6b5", target_key not in shown_target_keys, 1 if index % 2 == 0 else -1,
                            )
                    if is_overlap:
                        _add_link(
                            gene_node_id, "tumour:summary",
                            "Exact gene-symbol overlap in tumour-labelled summary",
                            f"The exact normalized symbol {symbol} is present in the indexed OpenPedCan gene-associated summary for tumour_type={chosen_disease!r}, with {observations:,} aggregated observations. The edge means a shared symbol in source records, not a causal or therapeutic relationship.",
                            "observed_overlap", "#e5bd76", True, 1 if index % 2 == 0 else -1,
                        )

                if additional_gene_count:
                    _add_node(
                        "gene:additional", f"{additional_gene_count} additional target-associated symbols",
                        f"+{additional_gene_count}\nMORE SYMBOLS", "Collapsed gene-symbol annotations",
                        "#8e83ad", 115, -300, -120,
                        f"{additional_gene_count} additional gene symbols are present in the ChEMBL target annotation extract but hidden for graph readability. The total symbol count is shown in the graph statistics.",
                        "research_evidence.sqlite · chembl_target_evidence · gene_symbol",
                        radius=10, label_size=8,
                    )
            elif target_rows:
                _add_node(
                    "gap:genes", "Target rows without usable gene symbols",
                    "NO USABLE\nGENE SYMBOLS", "Missing gene-symbol mapping",
                    "#d28d8d", 100, 0, 0,
                    "Target evidence rows were found, but no usable gene symbols were extracted from the gene_symbol column.",
                    "research_evidence.sqlite · chembl_target_evidence · gene_symbol",
                    radius=14, label_size=8,
                )

            # Tumour and OpenPedCan branches are real data links by exact
            # source-label grouping; they are not ontology or causal assertions.
            _add_node(
                "tumour:selected", chosen_disease,
                f"{chosen_disease}\nOPENPEDCAN TUMOUR LABEL",
                "OpenPedCan tumour label", "#f1bf73", 390, 40, 10,
                f"Selected OpenPedCan tumour_type source label: {chosen_disease!r}. Clinical metadata contains {labelled_biospecimens:,} distinct sample IDs with this exact label. The research index contains {indexed_gene_count:,} gene symbols and {tumour_observation_count:,} aggregated gene-associated observations for the same exact text label.",
                "data/raw/openpedcan/openpedcan_phase1.csv · tumor_type; research_evidence.sqlite · mutation_gene_summary",
                radius=29, label_size=11, bold=True, label_side="top",
            )
            _add_node(
                "clinical:coverage", f"{labelled_biospecimens:,} labelled sample IDs",
                f"{labelled_biospecimens:,}\nLABELLED SAMPLE IDS",
                "Clinical sample-label coverage", "#79afe1", 390, 290, -45,
                f"Distinct sample_id count in OpenPedCan clinical metadata where tumor_type exactly matches {chosen_disease!r}. This is label coverage, not a patient outcome or treatment measure.",
                "data/raw/openpedcan/openpedcan_phase1.csv · sample_id / tumor_type",
                radius=21, label_size=9,
            )
            _add_link(
                "clinical:coverage", "tumour:selected",
                "Clinical label coverage",
                f"{labelled_biospecimens:,} distinct sample IDs carry the reported source label {chosen_disease!r}. The label is preserved as provided; the cross-dataset disease crosswalk is not approved.",
                "clinical_coverage", "#79afe1", False,
            )
            _add_node(
                "tumour:summary", f"{indexed_gene_count:,} genes · {tumour_observation_count:,} observations",
                f"{indexed_gene_count:,} GENES\n{tumour_observation_count:,} OBSERVATIONS",
                "OpenPedCan gene-associated summary", "#58c6b5", 650, 40, -35,
                f"The local research index contains {indexed_gene_count:,} distinct gene symbols and {tumour_observation_count:,} aggregated gene-associated observations where tumour_type exactly matches {chosen_disease!r}. The graph uses this summary only for exact symbol-overlap inspection.",
                "data/processed/graph_v3/research_evidence.sqlite · mutation_gene_summary",
                radius=23, label_size=9, label_side="top",
            )
            _add_link(
                "tumour:selected", "tumour:summary",
                "Exact tumour-label grouping in research index",
                f"The same exact tumour_type text label {chosen_disease!r} is used to query the mutation-gene summary. This is an index grouping, not a validated cross-source disease equivalence.",
                "tumour_grouping", "#58c6b5", False,
            )

            # Keep clinical and molecular source files visible as provenance labels.
            _add_node(
                "source:clinical", "OpenPedCan clinical metadata",
                "CLINICAL SOURCE\nSAMPLE_ID / TUMOR_TYPE", "Source dataset",
                "#668eae", 120, 350, -100,
                "Clinical source fields used here: sample_id and tumor_type. Only distinct sample IDs and aggregate tumour-label counts are displayed.",
                "data/raw/openpedcan/openpedcan_phase1.csv",
                radius=13, label_size=8,
            )
            _add_link(
                "source:clinical", "clinical:coverage",
                "Clinical source → cohort count",
                "Distinct sample_id values are counted from the clinical metadata for the selected tumour_type text label.",
                "provenance", "#79afe1", False,
            )
            _add_node(
                "source:mutation", "OpenPedCan mutation source",
                "MUTATION SOURCE\nAGGREGATED INDEX", "Source dataset",
                "#448f83", 650, -175, -100,
                "The gene-associated summary is derived from the OpenPedCan mutation source and indexed by tumour_type and gene_symbol. The graph shows aggregated counts, not individual variant interpretation.",
                "data/raw/openpedcan/openpedcan_mutations_phase2.csv → research_evidence.sqlite",
                radius=13, label_size=8,
            )
            _add_link(
                "source:mutation", "tumour:summary",
                "Mutation source → aggregated index",
                "Mutation records are summarised by reported tumour_type and gene_symbol in the local research evidence index.",
                "provenance", "#58c6b5", False,
            )

            # Independent experimental-model branch: never join its cancer
            # labels directly to the selected OpenPedCan tumour without review.
            _add_node(
                "gdsc2:assay", f"{response_count:,} GDSC2 response records",
                f"GDSC2 ASSAY\n{response_count:,} MEASUREMENTS",
                "In-vitro model assay dataset", "#9bb2c0", -385, -300, 0,
                f"GDSC2 contains {response_count:,} response rows for this GDSC drug ID/name query across the source's reported cancer-type labels. These are experimental model measurements, not patient outcomes.",
                "research_evidence.sqlite · gdsc_response",
                radius=19, label_size=9, label_side="bottom",
            )
            _add_link(
                "drug:selected", "gdsc2:assay",
                "Separate GDSC2 response query by drug ID / name",
                "This branch uses the selected GDSC drug record to retrieve in-vitro response measurements. It is deliberately separate from the OpenPedCan tumour observation branch.",
                "assay_branch", "#9bb2c0", True,
            )
            if not response_types.empty:
                response_rows = response_types.to_dict("records")
                for index, row in enumerate(response_rows):
                    model_label = _kg_clean(row.get("model_label")) or "Unlabelled"
                    try:
                        records_n = int(row.get("response_records") or 0)
                        cell_lines_n = int(row.get("cell_lines") or 0)
                    except (TypeError, ValueError):
                        records_n, cell_lines_n = 0, 0
                    model_node_id = f"gdsc2:model:{index}"
                    y = -285 + index * 65
                    _add_node(
                        model_node_id,
                        model_label,
                        f"{model_label}\n{cell_lines_n:,} LINES · {records_n:,} ROWS",
                        "GDSC2 reported model cancer-type label", "#8299a8",
                        -80, y, (index % 3 - 1) * 75,
                        f"GDSC2 reported_cancer_type label {model_label!r}: {records_n:,} response rows across {cell_lines_n:,} distinct cell lines for the selected drug. This label belongs to the GDSC2 model dataset and is not automatically equivalent to the selected OpenPedCan tumour label.",
                        "research_evidence.sqlite · gdsc_response · reported_cancer_type / cell_line",
                        radius=12, label_size=8,
                    )
                    _add_link(
                        "gdsc2:assay", model_node_id,
                        "Reported GDSC2 model-label grouping",
                        f"The GDSC2 rows are grouped by their own reported_cancer_type value {model_label!r}. No automatic crosswalk to OpenPedCan is applied.",
                        "model_grouping", "#9bb2c0", False, 1 if index % 2 == 0 else -1,
                    )

            # Relationship-source node and the molecule/target path.
            _add_node(
                "source:chembl", "ChEMBL target evidence extract",
                "ChEMBL SOURCE\nTARGET ANNOTATIONS", "Source dataset",
                "#8773ad", -120, 315, -80,
                f"The local ChEMBL extract contains {len(target_rows):,} rows for the selected GDSC drug ID/name query. Target type, mapping method and gene-symbol annotations are retained from the source extract.",
                "data/raw/chembl/gdsc2_chembl_target_evidence.csv → research_evidence.sqlite",
                radius=13, label_size=8,
            )
            if shown_target_keys:
                for target_key in shown_target_keys:
                    _add_link(
                        "source:chembl", f"target:{target_key}",
                        "ChEMBL source record provenance",
                        "Target entity and attached gene-symbol annotations are taken from the local ChEMBL evidence extract.",
                        "provenance", "#8773ad", True,
                    )

            if not overlap_symbols:
                _add_node(
                    "gap:overlap", "No exact symbol overlap returned",
                    "0 EXACT SYMBOL\nOVERLAPS", "Unmatched symbol comparison",
                    "#d28d8d", 650, -300, 30,
                    f"No exact normalized gene-symbol matches were returned between the selected compound's ChEMBL target annotations and the indexed OpenPedCan mutation-gene summary for {chosen_disease!r}. This is a result of this indexed comparison, not proof that no biological relationship exists.",
                    "Exact normalized symbol comparison · ChEMBL target evidence × mutation_gene_summary",
                    radius=16, label_size=8,
                )
                _add_link(
                    "tumour:summary", "gap:overlap",
                    "Exact symbol comparison returned zero rows",
                    "The selected target-associated symbol set was compared with the selected tumour label's indexed gene-symbol summary. No exact normalized matches were returned.",
                    "gap", "#d28d8d", True,
                )

            graph_stats = {
                "targetRows": f"{len(target_rows):,}",
                "targetGenes": f"{len(all_target_genes):,}",
                "overlapGenes": f"{len(overlap_symbols):,}",
                "responseRows": f"{response_count:,}",
            }
            render_evidence_network_3d(
                nodes,
                links,
                title="Compound-to-tumour evidence network",
                tumour_label=chosen_disease,
                compound_label=selected_drug_name,
                stats=graph_stats,
                height=1250,
            )

            st.markdown(
                '<div class="ea-callout">How to read the graph: follow the GDSC compound record to its ChEMBL molecule mapping, target annotation and gene-symbol records. A dashed amber link means the exact normalized symbol is present in the selected OpenPedCan tumour summary. This is descriptive source-record overlap—not proof of target engagement, gene dependency, causality, therapeutic fit or efficacy. The lower GDSC2 assay/model branch is separate; its reported cancer-type labels are not automatically crosswalked to OpenPedCan tumour labels.</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Inspect selected compound target annotations", expanded=False):
                if target_records.empty:
                    st.info("No ChEMBL target evidence rows were found for this selected GDSC drug record. The graph shows the missing mapping rather than inventing a target.")
                else:
                    columns = [c for c in [
                        "molecule_chembl_id", "drug_match_method", "target_chembl_id",
                        "target_name", "target_type", "gene_symbol", "mechanism_of_action",
                        "action_type", "direct_interaction", "source_url",
                    ] if c in target_records.columns]
                    st.dataframe(target_records[columns].head(120), use_container_width=True, hide_index=True, height=300)
                    st.caption("For PROTEIN COMPLEX records, multiple gene symbols may denote complex components rather than independent drug targets.")

            with st.expander("Inspect exact gene-symbol overlap with selected tumour label"):
                if overlap.empty:
                    st.info("No exact normalized gene-symbol overlap was returned for this selection. This is not proof of biological absence.")
                else:
                    st.dataframe(
                        overlap.rename(columns={
                            "gene_symbol": "Gene symbol",
                            "observations": "OpenPedCan gene-associated observations",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            with st.expander("Inspect GDSC2 response summary — separate model evidence"):
                st.metric("GDSC2 response records for selected drug", f"{response_count:,}")
                if not response_types.empty:
                    st.dataframe(response_types.rename(columns={
                        "model_label": "GDSC2 reported cancer-type label",
                        "response_records": "Response records",
                        "cell_lines": "Distinct cell lines",
                    }), use_container_width=True, hide_index=True)
                st.caption("GDSC2 measurements are in-vitro model data. They are not patient outcomes and are not connected to OpenPedCan tumour labels by an assumed disease equivalence.")

            with st.expander("Suggested presentation narration"):
                st.write(
                    "Start at the selected GDSC compound record and follow its recorded ChEMBL molecule mapping, target annotations and gene-symbol labels. Then inspect which exact symbols appear in the selected OpenPedCan tumour-labelled mutation summary. The dashed edge represents descriptive symbol overlap, not causality. The GDSC2 assay branch is shown separately because cell-line model labels and patient/biospecimen tumour labels are not automatically equivalent. Every node and relationship can be inspected for its source."
                )

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
elif page == "Data integrity":
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
# Molecular evidence workbench: presentation-ready, live local evidence workflow.
# -----------------------------------------------------------------------------
elif page == "Molecular evidence workbench":
    st.markdown(
        '<div class="ea-workbench-head"><div><div class="ea-eyebrow">MOLECULAR EVIDENCE WORKBENCH · LIVE LOCAL DATA</div>'
        '<h2>Build an evidence dossier, one source at a time.</h2>'
        '<p>Select a compound, an OpenPedCan tumour label and a separate GDSC2 model label. The workspace shows what is recorded, what overlaps by exact identifier, and what remains unknown.</p></div>'
        '<div class="ea-workbench-tag">EXPLORATORY · NOT PREDICTIVE</div></div>',
        unsafe_allow_html=True,
    )

    if not research_ready:
        st.warning("The local research evidence index is not ready. Build and validate it before the live demo.")
        st.code("./.venv/bin/python scripts/build_research_evidence_index.py\n./.venv/bin/python scripts/validate_research_evidence_index.py", language="bash")
    else:
        try:
            demo_drugs = get_drugs().drop_duplicates(subset=["drug_id", "drug_name"]).copy()
            demo_gdsc_types = get_gdsc_cancer_types()
        except (sqlite3.Error, OSError) as exc:
            demo_drugs, demo_gdsc_types = pd.DataFrame(), []
            st.error(f"Could not read the research index: {exc}")

        if demo_drugs.empty:
            st.warning("No GDSC2 drug records are available in the research index.")
        else:
            demo_drugs["label"] = demo_drugs.apply(
                lambda r: f"{r['drug_name']} · GDSC ID {r['drug_id']}", axis=1
            )
            demo_labels = demo_drugs["label"].tolist()
            preferred_drug = next(
                (label for label in demo_labels if label.split(" · ")[0].strip().casefold() == "bortezomib"),
                demo_labels[0],
            )
            disease_counts = pd.Series(dtype="int64")
            if not clinical.empty and "tumor_type" in clinical and "sample_id" in clinical:
                disease_counts = (
                    clinical.dropna(subset=["tumor_type"])
                    .assign(tumor_type=lambda d: d["tumor_type"].astype(str).str.strip())
                    .query("tumor_type != ''")
                    .groupby("tumor_type")["sample_id"].nunique()
                    .sort_values(ascending=False)
                )
            disease_options = disease_counts.index.astype(str).tolist()
            if not disease_options:
                disease_options = ["Not available"]
            preferred_disease = next(
                (d for d in disease_options if d.casefold() == "neuroblastoma"),
                disease_options[0],
            )
            gdsc_options = ["All reported cancer types", *demo_gdsc_types]

            st.markdown('<div class="ea-mini-label">Choose a live data slice</div>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns([1.05, 1.05, 1.0])
            with f1:
                chosen_drug_label = st.selectbox(
                    "Drug record", demo_labels,
                    index=demo_labels.index(preferred_drug),
                    key="ea_demo_drug",
                )
            chosen_drug = demo_drugs.loc[demo_drugs["label"] == chosen_drug_label].iloc[0]
            with f2:
                chosen_disease = st.selectbox(
                    "OpenPedCan tumour label", disease_options,
                    index=disease_options.index(preferred_disease),
                    key="ea_demo_disease",
                )
            with f3:
                chosen_gdsc_type = st.selectbox(
                    "GDSC2 model label (separate source)", gdsc_options,
                    index=0, key="ea_demo_gdsc_type",
                    help="This is a separate cell-line dataset label. Selecting a label does not declare it equivalent to the OpenPedCan tumour label.",
                )

            drug_id = str(chosen_drug["drug_id"])
            drug_name = str(chosen_drug["drug_name"])
            target_records = get_chembl_targets(drug_id, drug_name)
            gene_values = []
            if not target_records.empty and "gene_symbol" in target_records:
                for raw_value in target_records["gene_symbol"].dropna().astype(str):
                    gene_values.extend(
                        part.strip().upper()
                        for part in raw_value.replace(";", ",").split(",")
                        if part.strip()
                    )
            target_genes = tuple(sorted({g for g in gene_values if g not in {"", "NAN", "UNKNOWN"}}))
            overlap = get_target_gene_overlap(chosen_disease, target_genes) if target_genes else pd.DataFrame(columns=["gene_symbol", "observations"])
            tumour_samples = int(disease_counts.get(chosen_disease, 0)) if not disease_counts.empty else 0
            response_count = get_drug_response_count(drug_id, drug_name, chosen_gdsc_type)
            response_frame = get_response_measurements(drug_id, drug_name, chosen_gdsc_type, max_rows=12000)
            if not response_frame.empty:
                for col in ("ln_ic50", "auc", "rmse"):
                    if col in response_frame:
                        response_frame[col] = pd.to_numeric(response_frame[col], errors="coerce")

            target_ids = target_records["target_chembl_id"].dropna().astype(str).nunique() if not target_records.empty and "target_chembl_id" in target_records else 0
            molecule_ids = target_records["molecule_chembl_id"].dropna().astype(str).nunique() if not target_records.empty and "molecule_chembl_id" in target_records else 0
            cell_lines = response_frame["cell_line"].nunique() if not response_frame.empty and "cell_line" in response_frame else 0
            median_ln_ic50 = response_frame["ln_ic50"].median() if not response_frame.empty and "ln_ic50" in response_frame else float("nan")
            median_auc = response_frame["auc"].median() if not response_frame.empty and "auc" in response_frame else float("nan")
            median_ln_ic50_label = f"{median_ln_ic50:.3f}" if pd.notna(median_ln_ic50) else "—"

            k1, k2, k3, k4 = st.columns(4)
            with k1: st.metric("ChEMBL target IDs", f"{target_ids:,}", help="Distinct target identifiers in the local ChEMBL extract for this drug mapping.")
            with k2: st.metric("Gene symbols in mapping", f"{len(target_genes):,}", help="Symbols attached to source target records; members of a protein complex may be represented separately.")
            with k3: st.metric("Exact symbol overlaps", f"{len(overlap):,}", help="Descriptive overlap with gene-associated OpenPedCan mutation observations; not evidence of dependency.")
            with k4: st.metric("GDSC2 response records", f"{response_count:,}", help="In-vitro cell-line measurements, not patient outcomes.")

            section("Follow the evidence", "Read the panels left to right. The connection is a research workflow, not a causal chain.")
            a, arrow1, b, arrow2, c = st.columns([1, .08, 1, .08, 1])
            with a:
                st.markdown(
                    '<div class="ea-demo-card"><div class="ea-demo-index">01 / SOURCE ANNOTATION</div>'
                    '<h3>What do the target records say?</h3>'
                    f'<div class="value">{len(target_records):,}</div>'
                    '<p>Rows in the local ChEMBL mapping extract for this selected GDSC drug record.</p></div>',
                    unsafe_allow_html=True,
                )
            with arrow1:
                st.markdown('<div class="ea-demo-arrow">→</div>', unsafe_allow_html=True)
            with b:
                st.markdown(
                    '<div class="ea-demo-card blue"><div class="ea-demo-index">02 / TUMOUR OBSERVATIONS</div>'
                    f'<h3>{chosen_disease}</h3><div class="value">{tumour_samples:,}</div>'
                    f'<p>Unique sample IDs with this tumour label in the OpenPedCan clinical metadata. {len(overlap):,} target-associated gene symbols overlap the indexed gene-associated variant summary.</p></div>',
                    unsafe_allow_html=True,
                )
            with arrow2:
                st.markdown('<div class="ea-demo-arrow">→</div>', unsafe_allow_html=True)
            with c:
                st.markdown(
                    '<div class="ea-demo-card violet"><div class="ea-demo-index">03 / EXPERIMENTAL CONTEXT</div>'
                    f'<h3>GDSC2 in-vitro assays</h3><div class="value">{response_count:,}</div>'
                    f'<p>{cell_lines:,} cell lines in the displayed query page; median ln(IC50) {median_ln_ic50_label}. Not a clinical outcome or a paediatric efficacy estimate.</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="ea-boundary-grid">'
                '<div class="ea-boundary-card"><h4>HOW A RESEARCHER CAN USE THIS VIEW</h4><p>Quickly inspect whether source target annotations exist, whether exact gene symbols occur in the selected tumour-labelled observations, and whether a separate experimental model dataset contains measurements to explore.</p></div>'
                '<div class="ea-boundary-card"><h4>WHAT THIS VIEW DOES NOT ESTABLISH</h4><p>It does not prove target engagement in tumour tissue, disease equivalence across datasets, gene dependency, causal mechanism, paediatric efficacy or patient benefit.</p></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Detail beneath the high-level visual journey.
            detail_left, detail_right = st.columns([1.05, 1])
            with detail_left:
                section("Source target evidence", "Provenance first; source annotations are not proof of activity in a tumour.")
                if target_records.empty:
                    st.info("No ChEMBL target rows were found for this exact drug mapping. Missing connections are left missing.")
                else:
                    view_cols = [c for c in ["molecule_chembl_id", "target_chembl_id", "target_name", "target_type", "gene_symbol", "mechanism_of_action", "drug_match_method"] if c in target_records.columns]
                    st.dataframe(target_records[view_cols].drop_duplicates().head(60), use_container_width=True, hide_index=True, height=260)
                    st.caption("Gene symbols can represent components of a protein complex; do not interpret each symbol as an independent target.")
            with detail_right:
                section("Observed gene-symbol overlap", "Only exact normalized symbols are counted; no pathway or causal edge is inferred.")
                if not overlap.empty and PLOTLY_AVAILABLE:
                    overlap_plot = overlap.head(12).sort_values("observations")
                    fig = px.bar(
                        overlap_plot, x="observations", y="gene_symbol", orientation="h",
                        color_discrete_sequence=[PALETTE["teal"]],
                        labels={"observations": "Gene-associated variant observations", "gene_symbol": "Gene symbol"},
                    )
                    style_figure(fig, 300)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                elif not overlap.empty:
                    st.dataframe(overlap.head(12), use_container_width=True, hide_index=True)
                else:
                    st.info("No exact gene-symbol overlap was found for this selected label and target mapping. This does not prove biological absence.")

            section("Experimental response distribution", "Descriptive cell-line assay measurements from GDSC2, kept separate from OpenPedCan patient/biospecimen data.")
            if response_frame.empty:
                st.info("No GDSC2 measurements were returned for the selected drug and model-label filter.")
            else:
                r1, r2, r3 = st.columns(3)
                with r1: st.metric("Cell lines in displayed rows", f"{cell_lines:,}")
                with r2: st.metric("Median ln(IC50)", f"{median_ln_ic50:.3f}" if pd.notna(median_ln_ic50) else "—")
                with r3: st.metric("Median AUC", f"{median_auc:.3f}" if pd.notna(median_auc) else "—")
                h1, h2 = st.columns(2)
                with h1:
                    if PLOTLY_AVAILABLE and response_frame["ln_ic50"].notna().any():
                        fig = px.histogram(response_frame.dropna(subset=["ln_ic50"]), x="ln_ic50", nbins=28, color_discrete_sequence=[PALETTE["blue"]], labels={"ln_ic50": "ln(IC50)"})
                        style_figure(fig, 290)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("ln(IC50) distribution is unavailable for this selection.")
                with h2:
                    if PLOTLY_AVAILABLE and response_frame["auc"].notna().any():
                        fig = px.histogram(response_frame.dropna(subset=["auc"]), x="auc", nbins=28, color_discrete_sequence=[PALETTE["violet"]], labels={"auc": "AUC"})
                        style_figure(fig, 290)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("AUC distribution is unavailable for this selection.")
                st.caption("The charts use up to 12,000 returned rows for responsiveness; the record count above is the exact count for the selected filter.")
                with st.expander("Inspect underlying GDSC2 rows"):
                    table_cols = [c for c in ["cell_line", "sanger_model_id", "reported_cancer_type", "ln_ic50", "auc", "rmse", "putative_target", "pathway"] if c in response_frame.columns]
                    st.dataframe(response_frame[table_cols].head(250), use_container_width=True, hide_index=True, height=300)

            st.markdown(
                '<div class="ea-callout">Interpretation boundary: source target annotations, gene-associated variant observations and cell-line responses are different evidence types. Exact overlap does not establish target dependency, biological mechanism, therapeutic fit or patient benefit. No drug-efficacy score is calculated.</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Suggested 60-second presentation narration"):
                st.write(
                    "Evidence Atlas is designed to make the path between research sources inspectable. "
                    "Here, target annotations, tumour-associated molecular observations and cell-line assay data are displayed in separate evidence lanes, each with its own provenance and denominator. "
                    "The prototype deliberately shows unresolved links and missing evidence instead of turning them into a confident-looking score. "
                    "The next milestone is expert review and independent validation—not a claim that the current interface predicts treatment response."
                )

# -----------------------------------------------------------------------------
# Molecular response model: held-out benchmark and cell-line exploration.
# -----------------------------------------------------------------------------
elif page == "Molecular response model":
    section(
        "Single-agent molecular-response benchmark",
        "Expression-based estimates for known GDSC2 drugs, evaluated on held-out cell lines.",
    )
    st.markdown(
        '<div class="ea-callout ea-callout-blue">This is an experimental cell-line research model, not a clinical or patient-response predictor. It does not predict unseen drugs or drug combinations. Molecular-model performance must be judged against the response-only baselines shown here.</div>',
        unsafe_allow_html=True,
    )

    required_artifacts = [
        MOLECULAR_MODEL_PATH,
        MOLECULAR_MODEL_SUMMARY_PATH,
        *MOLECULAR_PREDICTIONS.values(),
    ]
    artifacts_ready = all(path.is_file() for path in required_artifacts)
    if not artifacts_ready:
        missing_artifacts = [
            str(path.relative_to(PROJECT_ROOT))
            for path in required_artifacts
            if not path.is_file()
        ]
        st.warning(
            "The molecular model has not been trained in this workspace yet. "
            "Training streams the public E-MTAB-2770 matrix and may take several minutes."
        )
        st.caption("Missing output artifacts: " + ", ".join(missing_artifacts))
        st.code(
            "./.venv/bin/python scripts/ai/train_gdsc2_expression_model.py",
            language="bash",
        )
        if st.button(
            "Run molecular benchmark",
            type="primary",
            key="run_gdsc2_expression_benchmark",
        ):
            try:
                with st.spinner(
                    "Training and evaluating on held-out cell lines. "
                    "The first run downloads and streams the full expression matrix."
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts/ai/train_gdsc2_expression_model.py"),
                        ],
                        cwd=PROJECT_ROOT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                        capture_output=True,
                        text=True,
                        timeout=1800,
                        check=False,
                    )
                if result.returncode:
                    st.error(
                        "Molecular benchmark failed "
                        f"(exit code {result.returncode}). Review the run output below."
                    )
                    st.code(result.stderr or result.stdout or "No process output.")
                else:
                    st.success("Molecular benchmark completed and artifacts were written.")
                    if result.stdout:
                        with st.expander("Training log"):
                            st.code(result.stdout)
                    st.rerun()
            except subprocess.TimeoutExpired as exc:
                st.error(
                    "Molecular benchmark exceeded the 30-minute dashboard limit. "
                    "No partial model is treated as ready."
                )
                output = exc.stderr or exc.stdout
                if output:
                    st.code(output)
            except OSError as exc:
                st.error(f"Could not start the molecular benchmark: {exc}")

        st.markdown(
            '<div class="ea-panel"><div class="ea-panel-title">What this model can evaluate</div>'
            '<div class="ea-panel-sub">The pipeline uses exact cell-line-name matches, '
            'the frozen five-fold Sanger-ID split, training-fold gene selection, and '
            'per-drug Ridge regressions for GDSC2 LN(IC50) and AUC. It compares all '
            'model metrics on the same eligible held-out rows as response-only baselines.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            model_summary = json.loads(
                MOLECULAR_MODEL_SUMMARY_PATH.read_text(encoding="utf-8")
            )
            model_bundle = load_expression_model(MOLECULAR_MODEL_PATH)
            prediction_frames = {
                endpoint: pd.read_csv(path)
                for endpoint, path in zip(
                    ("ln_ic50", "auc"), MOLECULAR_PREDICTIONS.values()
                )
            }
        except (OSError, json.JSONDecodeError, ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not load the molecular model artifacts: {exc}")
        else:
            crosswalk = model_summary.get("crosswalk", {})
            coverage_columns = st.columns(4)
            coverage_columns[0].metric(
                "Matched cell lines",
                f"{crosswalk.get('matched_models', 0):,}",
                f"of {crosswalk.get('response_models', 0):,} GDSC2 models",
            )
            coverage_columns[1].metric(
                "Held-out response rows",
                f"{model_summary.get('matched_response_rows', 0):,}",
            )
            coverage_columns[2].metric(
                "Drugs benchmarked",
                f"{model_summary.get('matched_drugs', 0):,}",
            )
            coverage_columns[3].metric(
                "Unmatched GDSC2 models",
                f"{crosswalk.get('unmatched_models', 0):,}",
            )
            st.caption(
                "Name-based coverage is limited; ambiguous matches are excluded. "
                "The reported cell-line estimates are not independent clinical evidence."
            )

            for endpoint, title in (
                ("ln_ic50", "LN(IC50)"),
                ("auc", "AUC"),
            ):
                st.subheader(f"{title} held-out benchmark")
                endpoint_metrics = model_summary.get("metrics", {}).get(endpoint, {})
                if not endpoint_metrics:
                    st.warning(f"No {title} benchmark metrics are present.")
                    continue
                metrics_frame = pd.DataFrame(
                    [
                        {"Model": "Expression Ridge", **endpoint_metrics.get("expression_ridge", {})},
                        *[
                            {"Model": name.replace("_", " ").title(), **values}
                            for name, values in endpoint_metrics.get(
                                "baselines_on_same_rows", {}
                            ).items()
                        ],
                    ]
                )
                st.dataframe(
                    metrics_frame,
                    use_container_width=True,
                    hide_index=True,
                )
                difference = endpoint_metrics.get(
                    "macro_drug_mae_difference_vs_drug_cancer_baseline"
                )
                if difference is not None:
                    if difference < 0:
                        st.success(
                            f"Expression model macro drug MAE is {abs(difference):.4f} "
                            "lower than the drug+cancer median baseline on these rows."
                        )
                    else:
                        st.warning(
                            f"Expression model macro drug MAE is {difference:.4f} "
                            "higher than the drug+cancer median baseline on these rows."
                        )

            st.subheader("Inspect held-out predictions")
            endpoint_label = st.selectbox(
                "Response endpoint",
                ["LN(IC50)", "AUC"],
                key="gdsc2_expression_endpoint",
            )
            endpoint_key = "ln_ic50" if endpoint_label == "LN(IC50)" else "auc"
            prediction_frame = prediction_frames[endpoint_key]
            drug_choices = (
                prediction_frame[["drug_id", "drug_name"]]
                .drop_duplicates()
                .sort_values(["drug_name", "drug_id"])
            )
            drug_options = {
                f"{row.drug_name} · GDSC {row.drug_id}": row.drug_id
                for row in drug_choices.itertuples(index=False)
                if prediction_frame.loc[
                    prediction_frame["drug_id"].astype(str) == str(row.drug_id),
                    "expression_ridge",
                ].notna().any()
            }
            if drug_options:
                chosen_drug_label = st.selectbox(
                    "Known drug",
                    list(drug_options),
                    key="gdsc2_expression_drug",
                )
                chosen_drug_id = str(drug_options[chosen_drug_label])
                selected_predictions = prediction_frame[
                    prediction_frame["drug_id"].astype(str) == chosen_drug_id
                ].copy()
                selected_predictions = selected_predictions.dropna(
                    subset=["expression_ridge"]
                )
                display_columns = [
                    "cell_line",
                    "sanger_model_id",
                    "reported_cancer_type",
                    "fold",
                    endpoint_key,
                    "expression_ridge",
                    "baseline_drug_cancer_median",
                ]
                st.dataframe(
                    selected_predictions[display_columns].rename(
                        columns={
                            endpoint_key: f"Observed {endpoint_label}",
                            "expression_ridge": "Expression estimate",
                            "baseline_drug_cancer_median": "Drug+cancer baseline",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )
            else:
                st.info("No trained per-drug models have eligible held-out predictions.")

            st.subheader("Explore a cell-line expression profile")
            st.write(
                "Upload a CSV with `gene_id` and raw `tpm` columns. The file must contain "
                "every gene feature required by the selected trained drug model."
            )
            profile_file = st.file_uploader(
                "Cell-line TPM profile",
                type=["csv"],
                key="gdsc2_expression_profile",
            )
            trained_drugs = {
                str(drug_id): details.get("drug_name", str(drug_id))
                for endpoint_models in model_bundle["targets"].values()
                for drug_id, details in endpoint_models.items()
            }
            if profile_file is not None and trained_drugs:
                selected_model_drug = st.selectbox(
                    "Drug for expression-profile estimate",
                    sorted(
                        trained_drugs,
                        key=lambda drug_id: trained_drugs[drug_id].casefold(),
                    ),
                    format_func=lambda drug_id: (
                        f"{trained_drugs[drug_id]} · GDSC {drug_id}"
                    ),
                    key="gdsc2_expression_profile_drug",
                )
                if st.button(
                    "Estimate cell-line response",
                    key="predict_gdsc2_expression_profile",
                ):
                    try:
                        profile = pd.read_csv(profile_file)
                        if not {"gene_id", "tpm"}.issubset(profile.columns):
                            raise ValueError("CSV must include `gene_id` and `tpm` columns.")
                        profile["gene_id"] = profile["gene_id"].fillna("").astype(str).str.strip()
                        if profile["gene_id"].eq("").any():
                            raise ValueError("CSV contains an empty gene_id.")
                        if profile["gene_id"].duplicated().any():
                            raise ValueError("CSV contains duplicate gene_id values.")
                        tpm_by_gene = dict(
                            zip(profile["gene_id"], pd.to_numeric(profile["tpm"], errors="raise"))
                        )
                        result = predict_expression_response(
                            model_bundle,
                            selected_model_drug,
                            tpm_by_gene,
                        )
                    except (OSError, pd.errors.ParserError, TypeError, ValueError, KeyError) as exc:
                        st.error(f"Could not estimate response from this profile: {exc}")
                    else:
                        st.json(result)
                        st.warning(
                            "These are model estimates for a cell-line expression profile "
                            "on GDSC2 endpoints. They are not patient-specific, clinical, "
                            "or combination-treatment predictions."
                        )
            elif profile_file is not None:
                st.info("No per-drug expression model is available for profile prediction.")

# -----------------------------------------------------------------------------
# Page 7: Investor briefing — truthful, stage-aware narrative
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
