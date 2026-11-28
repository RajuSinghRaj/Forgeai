"""Streamlit entry point for ForgeAI — The Agent That Builds Agents.

Phase 4 adds automatic evaluation-set generation (20+ cases), hybrid judging,
scorecard reporting, and evaluation artifact export while preserving Phase 1–3.
Importing or starting this app never triggers Gemini API calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import streamlit as st

from forgeai.config import load_settings
from forgeai.evaluation import build_evaluation_report_pdf_from_state
from forgeai.llm import GeminiClient
from forgeai.llm.telemetry import read_telemetry_summary
from forgeai.retrieval.loaders import load_document_folder
from forgeai.retrieval.uploads import UploadValidationError, prepare_uploaded_corpus
from forgeai.retrieval.vector_store import build_isolated_vector_store
from forgeai.runner import DeclarativeAgentRunner
from forgeai.schemas import BriefInput, GroundabilityVerdict
from forgeai.tools import build_default_registry
from forgeai.workflow import build_factory_graph, list_saved_runs, load_run_checkpoint
from forgeai.workflow.protection import (
    compute_evaluation_signature,
    execution_lock_guard,
    is_evaluation_already_completed,
    is_optimization_already_completed,
)

st.set_page_config(
    page_title="ForgeAI — Agent Factory",
    page_icon="F",
    layout="wide",
)


def _render_glass_theme(dark_mode: bool = False) -> None:
    """Apply the ForgeAI pure glassmorphism visual system (Light & Dark)."""
    if dark_mode:
        palette = """
        :root {
            color-scheme: dark;
            --fg:            #f1f5f9;
            --fg-muted:      #cbd5e1;
            --fg-subtle:     #94a3b8;
            --bg-canvas:
              radial-gradient(ellipse 75% 55% at 12% 4%, rgba(79, 70, 229, 0.24) 0%, transparent 58%),
              radial-gradient(ellipse 60% 50% at 88% 12%, rgba(6, 182, 212, 0.18) 0%, transparent 55%),
              radial-gradient(ellipse 65% 55% at 52% 94%, rgba(16, 185, 129, 0.14) 0%, transparent 55%),
              linear-gradient(155deg, #090d16 0%, #0f172a 48%, #0b1120 100%);
            --glass-bg:      linear-gradient(145deg, rgba(30, 41, 59, 0.62) 0%, rgba(15, 23, 42, 0.46) 100%);
            --glass-card:    linear-gradient(145deg, rgba(30, 41, 59, 0.72) 0%, rgba(15, 23, 42, 0.52) 100%);
            --glass-hero:
              radial-gradient(ellipse 70% 60% at 50% 0%, rgba(56, 189, 248, 0.12) 0%, transparent 55%),
              linear-gradient(145deg, rgba(30, 41, 59, 0.68) 0%, rgba(15, 23, 42, 0.56) 100%);
            --glass-sidebar: linear-gradient(175deg, rgba(15, 23, 42, 0.78) 0%, rgba(15, 23, 42, 0.64) 100%);
            --glass-border:  rgba(255, 255, 255, 0.14);
            --glass-top:     rgba(255, 255, 255, 0.24);
            --glass-inset:   rgba(255, 255, 255, 0.09);
            --glass-shadow:  0 18px 48px rgba(0, 0, 0, 0.42);
            --input-bg:      rgba(15, 23, 42, 0.62);
            --input-border:  rgba(148, 163, 184, 0.25);
            --placeholder:   #64748b;
            --sec-btn-bg:    rgba(30, 41, 59, 0.65);
            --sec-btn-hover: rgba(51, 65, 85, 0.85);
            --sec-btn-bdr:   rgba(255, 255, 255, 0.18);
            --accent:        #6366f1;
            --accent-2:      #06b6d4;
            --emerald:       #34d399;
            --code-bg:       rgba(99, 102, 241, 0.18);
            --code-fg:       #c7d2fe;
            --divider:       rgba(255, 255, 255, 0.10);
            --geo-stroke:    rgba(148, 163, 184, 0.28);
        }
        """
    else:
        palette = """
        :root {
            color-scheme: light;
            --fg:            #0f172a;
            --fg-muted:      #475569;
            --fg-subtle:     #64748b;
            --bg-canvas:
              radial-gradient(ellipse 75% 55% at 10% 2%, rgba(99, 102, 241, 0.13) 0%, transparent 56%),
              radial-gradient(ellipse 60% 50% at 90% 14%, rgba(6, 182, 212, 0.11) 0%, transparent 54%),
              radial-gradient(ellipse 65% 55% at 50% 94%, rgba(52, 211, 153, 0.11) 0%, transparent 56%),
              linear-gradient(155deg, #eef2ff 0%, #f8fafc 46%, #ecfeff 100%);
            --glass-bg:      linear-gradient(145deg, rgba(255, 255, 255, 0.68) 0%, rgba(255, 255, 255, 0.38) 100%);
            --glass-card:    linear-gradient(145deg, rgba(255, 255, 255, 0.78) 0%, rgba(255, 255, 255, 0.45) 100%);
            --glass-hero:
              radial-gradient(ellipse 70% 58% at 50% 0%, rgba(255, 255, 255, 0.94) 0%, rgba(248, 250, 252, 0.72) 55%, rgba(238, 242, 255, 0.48) 100%);
            --glass-sidebar: linear-gradient(175deg, rgba(255, 255, 255, 0.72) 0%, rgba(241, 245, 249, 0.56) 100%);
            --glass-border:  rgba(255, 255, 255, 0.85);
            --glass-top:     rgba(255, 255, 255, 0.98);
            --glass-inset:   rgba(255, 255, 255, 0.90);
            --glass-shadow:  0 16px 44px rgba(15, 23, 42, 0.08);
            --input-bg:      rgba(255, 255, 255, 0.62);
            --input-border:  rgba(148, 163, 184, 0.34);
            --placeholder:   #94a3b8;
            --sec-btn-bg:    rgba(255, 255, 255, 0.62);
            --sec-btn-hover: rgba(255, 255, 255, 0.92);
            --sec-btn-bdr:   rgba(99, 102, 241, 0.22);
            --accent:        #4f46e5;
            --accent-2:      #06b6d4;
            --emerald:       #10b981;
            --code-bg:       rgba(79, 70, 229, 0.09);
            --code-fg:       #4338ca;
            --divider:       rgba(15, 23, 42, 0.09);
            --geo-stroke:    rgba(100, 116, 139, 0.28);
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        {palette}

        *, *::before, *::after {{ box-sizing: border-box; }}
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif !important;
            color: var(--fg) !important;
        }}
        [data-testid="stAppViewContainer"] {{
            background: var(--bg-canvas) !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
            backdrop-filter: blur(12px) !important;
        }}
        #MainMenu, footer {{ visibility: hidden; }}

        /* ── Pure Glass Sidebar ── */
        [data-testid="stSidebar"] {{
            background: var(--glass-sidebar) !important;
            border-right: 1px solid var(--glass-border) !important;
            backdrop-filter: blur(28px) saturate(150%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(150%) !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {{
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--accent) !important;
            font-weight: 700 !important;
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--fg-muted) !important;
            font-size: 0.82rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: var(--fg-subtle) !important;
            font-size: 0.75rem !important;
        }}

        [data-testid="stMain"] > div {{
            padding: 0.5rem clamp(0.75rem, 3vw, 3rem) 3rem !important;
        }}

        /* ── Typography ── */
        h1, h2, h3, h4 {{
            color: var(--fg) !important;
            letter-spacing: -0.03em !important;
        }}
        p, li {{ color: var(--fg-muted) !important; }}
        strong {{ color: var(--fg) !important; }}
        [data-testid="stCaptionContainer"] {{ color: var(--fg-subtle) !important; }}
        [data-testid="stWidgetLabel"] p, label, legend {{
            color: var(--fg) !important;
            font-weight: 600 !important;
        }}

        /* ── Pure Frosted Glass Containers ── */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stDataFrame"],
        [data-testid="stAlert"] {{
            background: var(--glass-bg) !important;
            border: 1px solid var(--glass-border) !important;
            border-top-color: var(--glass-top) !important;
            border-radius: 20px !important;
            box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-inset) !important;
            backdrop-filter: blur(26px) saturate(155%) !important;
            -webkit-backdrop-filter: blur(26px) saturate(155%) !important;
        }}
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {{
            border-style: dashed !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploader"] small {{
            color: var(--fg-muted) !important;
            opacity: 1 !important;
        }}
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {{
            background: transparent !important;
            color: var(--fg) !important;
            font-weight: 600 !important;
        }}
        [data-testid="stAlert"] > div,
        [data-testid="stAlertContainer"],
        [data-testid="stAlert"] [role="alert"],
        [data-testid="stAlert"] [data-baseweb="notification"] {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--fg) !important;
        }}
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {{
            color: var(--fg) !important;
        }}

        /* ── Metric Tiles ── */
        [data-testid="stMetric"] {{
            background: var(--glass-card) !important;
            border: 1px solid var(--glass-border) !important;
            border-top-color: var(--glass-top) !important;
            border-radius: 18px !important;
            padding: 16px 20px !important;
            box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-inset) !important;
            backdrop-filter: blur(22px) saturate(150%) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--fg-subtle) !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--fg) !important;
            font-weight: 800 !important;
        }}

        /* ── Inputs (override Baseweb dark wrappers so typed text is always crisp) ── */
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stTextArea"] [data-baseweb="base-input"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] [data-baseweb="base-input"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background: var(--input-bg) !important;
            background-color: var(--input-bg) !important;
            color: var(--fg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 14px !important;
            backdrop-filter: blur(14px) !important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {{
            background: transparent !important;
            background-color: transparent !important;
            color: var(--fg) !important;
            -webkit-text-fill-color: var(--fg) !important;
            caret-color: var(--fg) !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] span {{
            color: var(--fg) !important;
        }}
        input::placeholder, textarea::placeholder {{
            color: var(--placeholder) !important;
            -webkit-text-fill-color: var(--placeholder) !important;
            opacity: 1 !important;
        }}
        [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.22) !important;
        }}

        /* ── Buttons (Vibrant Glassmorphic Gradient — No Black Button!) ── */
        .stButton > button,
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stBaseButton-secondary"] {{
            border-radius: 999px !important;
            border: 1px solid var(--sec-btn-bdr) !important;
            background: var(--sec-btn-bg) !important;
            color: var(--fg) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            min-height: 2.6rem !important;
            padding: 0 1.45rem !important;
            backdrop-filter: blur(16px) !important;
            box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-inset) !important;
            transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease !important;
        }}
        .stButton > button:hover:not(:disabled),
        [data-testid="stDownloadButton"] button:hover:not(:disabled),
        [data-testid="stFormSubmitButton"] button:hover:not(:disabled),
        [data-testid="stFileUploader"] button:hover:not(:disabled),
        [data-testid="stBaseButton-secondary"]:hover:not(:disabled) {{
            transform: translateY(-2px) !important;
            background: var(--sec-btn-hover) !important;
            box-shadow: 0 10px 28px rgba(79, 70, 229, 0.20) !important;
        }}
        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {{
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
            box-shadow: 0 8px 26px rgba(79, 70, 229, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.32) !important;
        }}
        .stButton > button[kind="primary"] p,
        [data-testid="stFormSubmitButton"] button[kind="primary"] p {{
            color: #ffffff !important;
        }}
        .stButton > button[kind="primary"]:hover:not(:disabled),
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover:not(:disabled) {{
            background: linear-gradient(135deg, #4338ca 0%, #0891b2 100%) !important;
            box-shadow: 0 12px 34px rgba(6, 182, 212, 0.36) !important;
        }}
        .stButton > button:disabled,
        [data-testid="stFormSubmitButton"] button:disabled {{
            opacity: 0.55 !important;
            cursor: not-allowed !important;
        }}

        [data-testid="stCode"] {{
            background: var(--input-bg) !important;
            color: var(--fg) !important;
            border-radius: 12px !important;
            border: 1px solid var(--glass-border) !important;
        }}
        hr {{ border-color: var(--divider) !important; }}

        /* ── Hero Section (Nunito-inspired Pure Glassmorphism for ForgeAI) ── */
        .forge-hero {{
            position: relative;
            overflow: hidden;
            padding: 1.75rem 2.2rem 2.2rem;
            border-radius: 28px;
            margin-bottom: 1.6rem;
            border: 1px solid var(--glass-border);
            border-top-color: var(--glass-top);
            background: var(--glass-hero);
            box-shadow: var(--glass-shadow), inset 0 1.5px 0 var(--glass-inset);
            backdrop-filter: blur(32px) saturate(160%);
            -webkit-backdrop-filter: blur(32px) saturate(160%);
        }}
        .forge-orb-bg-1 {{
            position: absolute; top: -70px; left: -50px; width: 280px; height: 280px;
            border-radius: 50%; pointer-events: none;
            background: radial-gradient(circle, rgba(79, 70, 229, 0.18) 0%, transparent 70%);
        }}
        .forge-orb-bg-2 {{
            position: absolute; bottom: -90px; right: -40px; width: 320px; height: 320px;
            border-radius: 50%; pointer-events: none;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.16) 0%, transparent 70%);
        }}
        .forge-geo-tl {{
            position: absolute; bottom: 20px; left: 26px; opacity: 0.65; pointer-events: none;
        }}
        .forge-geo-br {{
            position: absolute; bottom: 20px; right: 26px; opacity: 0.65; pointer-events: none;
        }}
        .forge-nav {{
            position: relative; z-index: 2;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
        }}
        .forge-brand {{
            display: flex; align-items: center; gap: 10px;
            font-size: 1.12rem; font-weight: 800; letter-spacing: -0.04em; color: var(--fg);
        }}
        .forge-mark {{
            display: inline-grid; place-items: center;
            width: 34px; height: 34px; border-radius: 12px 12px 12px 4px;
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            color: #ffffff; font-size: 0.92rem; font-weight: 800;
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
        }}
        .forge-nav-pills {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .forge-nav-pill {{
            padding: 5px 13px; border-radius: 999px;
            border: 1px solid var(--glass-border);
            background: var(--glass-card);
            color: var(--fg-muted); font-size: 0.74rem; font-weight: 600;
            backdrop-filter: blur(12px);
        }}
        .forge-hero-center {{
            position: relative; z-index: 2;
            text-align: center; margin: 1.8rem auto 0; max-width: 920px;
        }}
        .forge-headline {{
            font-size: clamp(2.5rem, 6vw, 4.7rem);
            font-weight: 800; line-height: 1.02; letter-spacing: -0.055em;
            color: var(--fg); margin: 0;
        }}
        .forge-headline-grad {{
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 55%, #10b981 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .forge-arrow-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            width: clamp(44px, 5vw, 62px); height: clamp(44px, 5vw, 62px);
            border-radius: 50%; vertical-align: middle; margin: 0 8px 6px;
            background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
            box-shadow: 0 10px 26px rgba(16, 185, 129, 0.32), inset 0 1.5px 0 rgba(255, 255, 255, 0.6);
        }}
        .forge-subcopy {{
            max-width: 610px; margin: 1.05rem auto 0;
            font-size: 0.98rem; line-height: 1.65; color: var(--fg-muted); font-weight: 500;
        }}

        /* ── Floating Agentic Glass Nodes ── */
        .forge-flow-grid {{
            position: relative; z-index: 2;
            display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px;
            margin: 1.9rem auto 0.2rem; max-width: 1020px;
        }}
        .forge-flow-card {{
            position: relative; text-align: left;
            padding: 15px 16px; border-radius: 18px;
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            border-top-color: var(--glass-top);
            box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-inset);
            backdrop-filter: blur(20px) saturate(150%);
        }}
        .forge-flow-top {{
            display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;
        }}
        .forge-flow-step {{
            font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.1em; color: var(--fg-subtle);
        }}
        .forge-flow-dot {{
            width: 9px; height: 9px; border-radius: 50%;
        }}
        .dot-indigo {{ background: #6366f1; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.22); }}
        .dot-cyan   {{ background: #06b6d4; box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.22); }}
        .dot-emerald{{ background: #10b981; box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.22); }}
        .dot-violet {{ background: #8b5cf6; box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.22); }}
        .forge-flow-title {{
            font-size: 0.92rem; font-weight: 700; color: var(--fg); margin-bottom: 4px;
        }}
        .forge-flow-desc {{
            font-size: 0.76rem; line-height: 1.45; color: var(--fg-subtle);
        }}

        /* ── Sidebar Glass Status Card ── */
        .forge-section-label {{
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.11em;
            text-transform: uppercase; color: var(--accent); margin: 0.5rem 0 0.45rem; display: block;
        }}
        .forge-sys-card {{
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            border-top-color: var(--glass-top);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-inset);
            backdrop-filter: blur(18px);
        }}
        .forge-sys-row {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 5px 0; border-bottom: 1px solid var(--divider); font-size: 0.78rem;
        }}
        .forge-sys-row:last-child {{ border-bottom: none; }}
        .forge-sys-key {{ color: var(--fg-subtle); font-weight: 500; }}
        .forge-sys-val {{ color: var(--fg); font-weight: 600; font-size: 0.76rem; }}
        .forge-sys-val code {{
            background: var(--code-bg); color: var(--code-fg);
            padding: 2px 7px; border-radius: 6px; font-size: 0.70rem;
        }}
        .forge-badge-ok {{
            display: inline-flex; align-items: center; gap: 5px;
            padding: 2px 9px; border-radius: 999px; font-size: 0.70rem; font-weight: 700;
            background: rgba(16, 185, 129, 0.14); color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.30);
        }}
        .forge-badge-err {{
            display: inline-flex; align-items: center; gap: 5px;
            padding: 2px 9px; border-radius: 999px; font-size: 0.70rem; font-weight: 700;
            background: rgba(244, 63, 94, 0.14); color: #f43f5e;
            border: 1px solid rgba(244, 63, 94, 0.30);
        }}

        @media (max-width: 960px) {{
            .forge-flow-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 640px) {{
            .forge-flow-grid {{ grid-template-columns: 1fr; }}
            .forge-nav-pills {{ display: none; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_CONFIRM_KEYS = (
    "confirm_intake_rerun",
    "confirm_agent_regen",
    "confirm_eval_gen_rerun",
    "confirm_eval_rerun",
    "confirm_opt_rerun",
)


def _init_session_state() -> None:
    defaults = {
        "factory_state": None,
        "last_error": None,
        "last_error_section": None,
        "active_operation": None,
        "reset_confirm_keys": False,
        "uploaded_corpus_meta": None,
        "chat_history": [],
        "chat_corpus_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("reset_confirm_keys"):
        st.session_state["reset_confirm_keys"] = False
        for confirm_key in _CONFIRM_KEYS:
            if confirm_key in st.session_state:
                st.session_state[confirm_key] = False


def _clear_error() -> None:
    st.session_state["last_error"] = None
    st.session_state["last_error_section"] = None


def _record_error_and_rerun(exc: Exception, *, section: str | None = None) -> None:
    """Persist error text in session state and trigger an immediate rerun so it renders."""
    st.session_state["last_error"] = str(exc)
    st.session_state["last_error_section"] = section
    st.rerun()


def _execute_guarded_action(
    operation_name: str,
    action_fn,
    *,
    section: str,
    spinner_text: str,
    mark_evaluated: bool = False,
    mark_optimized: bool = False,
    update_factory_state: bool = True,
    rerun: bool = True,
) -> None:
    """Run an expensive pipeline stage under a non-overlapping execution lock."""
    if st.session_state.get("active_operation"):
        st.session_state["last_error"] = (
            f"Operation '{st.session_state['active_operation']}' is already running. Wait until it completes."
        )
        st.session_state["last_error_section"] = section
        st.rerun()
        return

    _clear_error()
    need_rerun = False
    err_to_report: Exception | None = None

    with execution_lock_guard() as acquired:
        if not acquired:
            st.session_state["last_error"] = (
                "Another evaluation or optimization operation is already in progress. Please wait."
            )
            st.session_state["last_error_section"] = section
            need_rerun = True
        else:
            st.session_state["active_operation"] = operation_name
            try:
                with st.spinner(spinner_text):
                    result_state = action_fn()
                if update_factory_state and result_state is not None:
                    if mark_evaluated:
                        sig = compute_evaluation_signature(
                            result_state.get("agent_definition"),
                            result_state.get("eval_suite_sha256"),
                        )
                        if sig is not None:
                            result_state["evaluated_signature"] = sig
                    if mark_optimized:
                        sig = compute_evaluation_signature(
                            result_state.get("agent_definition"),
                            result_state.get("eval_suite_sha256"),
                        )
                        if sig is not None:
                            result_state["optimized_signature"] = sig
                            if result_state.get("eval_results"):
                                result_state["evaluated_signature"] = sig
                    st.session_state["factory_state"] = result_state
                st.session_state["reset_confirm_keys"] = True
                need_rerun = rerun
            except Exception as exc:
                err_to_report = exc
            finally:
                st.session_state["active_operation"] = None

    if err_to_report is not None:
        _record_error_and_rerun(err_to_report, section=section)
    elif need_rerun:
        st.rerun()


def main() -> None:
    _init_session_state()

    with st.sidebar:
        st.markdown("<span class='forge-section-label'>Appearance</span>", unsafe_allow_html=True)
        theme_choice = st.radio(
            "Theme",
            options=["Light Glass", "Dark Glass"],
            horizontal=True,
            key="ui_theme_choice",
            label_visibility="collapsed",
        )
    dark_mode = theme_choice == "Dark Glass"
    _render_glass_theme(dark_mode=dark_mode)

    settings = load_settings()
    status = settings.status_summary()
    ready, issues = settings.gemini_readiness()

    st.markdown(
        """
        <section class="forge-hero">
          <div class="forge-orb-bg-1" aria-hidden="true"></div>
          <div class="forge-orb-bg-2" aria-hidden="true"></div>
          <svg class="forge-geo-tl" width="54" height="54" viewBox="0 0 54 54" fill="none" aria-hidden="true">
            <polygon points="8,44 26,10 46,38" stroke="var(--geo-stroke)" stroke-width="1.4"/>
            <line x1="26" y1="10" x2="22" y2="46" stroke="var(--geo-stroke)" stroke-width="1.2"/>
            <circle cx="8" cy="44" r="2.2" fill="var(--geo-stroke)"/>
            <circle cx="26" cy="10" r="2.2" fill="var(--geo-stroke)"/>
            <circle cx="46" cy="38" r="2.2" fill="var(--geo-stroke)"/>
          </svg>
          <svg class="forge-geo-br" width="54" height="54" viewBox="0 0 54 54" fill="none" aria-hidden="true">
            <polygon points="10,14 44,18 28,46" stroke="var(--geo-stroke)" stroke-width="1.4"/>
            <line x1="10" y1="14" x2="36" y2="32" stroke="var(--geo-stroke)" stroke-width="1.2"/>
            <circle cx="10" cy="14" r="2.2" fill="var(--geo-stroke)"/>
            <circle cx="44" cy="18" r="2.2" fill="var(--geo-stroke)"/>
            <circle cx="28" cy="46" r="2.2" fill="var(--geo-stroke)"/>
          </svg>

          <div class="forge-nav">
            <div class="forge-brand">
              <span class="forge-mark">F</span>ForgeAI
            </div>
            <div class="forge-nav-pills">
              <span class="forge-nav-pill">Declarative Agents</span>
              <span class="forge-nav-pill">Grounded RAG</span>
              <span class="forge-nav-pill">Hybrid Judge</span>
              <span class="forge-nav-pill">Critic-Optimizer</span>
            </div>
          </div>

          <div class="forge-hero-center">
            <h1 class="forge-headline">
              Forge <span class="forge-headline-grad">agents</span>
              <span class="forge-arrow-badge" aria-hidden="true">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M5 12H19M19 12L13 6M19 12L13 18" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </span><br>
              that reason &amp; verify.
            </h1>
            <p class="forge-subcopy">
              Turn raw business briefs and multi-format documents into isolated, evaluated, self-optimizing AI agents with transparent citations.
            </p>
          </div>

          <div class="forge-flow-grid">
            <div class="forge-flow-card">
              <div class="forge-flow-top">
                <span class="forge-flow-step">01 · Intake</span>
                <span class="forge-flow-dot dot-indigo"></span>
              </div>
              <div class="forge-flow-title">Corpus &amp; Brief</div>
              <div class="forge-flow-desc">Isolated upload indexing across PDF, DOCX, CSV, TXT, and Markdown.</div>
            </div>
            <div class="forge-flow-card">
              <div class="forge-flow-top">
                <span class="forge-flow-step">02 · Verify</span>
                <span class="forge-flow-dot dot-emerald"></span>
              </div>
              <div class="forge-flow-title">Groundability Guard</div>
              <div class="forge-flow-desc">Detects unsupported scope and evidence gaps before generation.</div>
            </div>
            <div class="forge-flow-card">
              <div class="forge-flow-top">
                <span class="forge-flow-step">03 · Build</span>
                <span class="forge-flow-dot dot-cyan"></span>
              </div>
              <div class="forge-flow-title">Declarative Blueprint</div>
              <div class="forge-flow-desc">Synthesizes system prompt, retrieval config, tools, and output contract.</div>
            </div>
            <div class="forge-flow-card">
              <div class="forge-flow-top">
                <span class="forge-flow-step">04 · Refine</span>
                <span class="forge-flow-dot dot-violet"></span>
              </div>
              <div class="forge-flow-title">Eval &amp; Critic-Optimizer</div>
              <div class="forge-flow-desc">20+ generated cases, hybrid judge scoring, and targeted prompt tuning.</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.divider()
        st.markdown(
            f"""
            <span class="forge-section-label">System Status</span>
            <div class="forge-sys-card">
              <div class="forge-sys-row">
                <span class="forge-sys-key">Gemini API</span>
                <span class="{'forge-badge-ok' if ready else 'forge-badge-err'}">
                  {'Connected' if ready else 'Not configured'}
                </span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Model</span>
                <span class="forge-sys-val"><code>{status['gemini_model']}</code></span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Embeddings</span>
                <span class="forge-sys-val"><code>{status['embedding_model']}</code></span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">CSV rows / unit</span>
                <span class="forge-sys-val">{settings.csv_rows_per_unit}</span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Min eval cases</span>
                <span class="forge-sys-val">{settings.min_eval_cases}</span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Judge batch size</span>
                <span class="forge-sys-val">{settings.judge_batch_size}</span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Target pass rate</span>
                <span class="forge-sys-val">{settings.target_pass_rate:.0%}</span>
              </div>
              <div class="forge-sys-row">
                <span class="forge-sys-key">Max opt iterations</span>
                <span class="forge-sys-val">{settings.max_optimization_iterations}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not ready:
            st.warning("Gemini API is not yet configured for live runs.")
            for issue in issues:
                st.caption(f"- {issue}")

        telem = read_telemetry_summary(settings.artifacts_dir)
        st.divider()
        st.subheader("API Telemetry")
        st.caption(
            f"Requests logged: `{telem['total_requests']}` | Retries: `{telem['total_retries']}` | Errors: `{telem['total_errors']}`"
        )
        st.caption(
            f"Agent gen reqs: `{telem.get('agent_generation_requests', 0)}` (saved `{telem.get('agent_generation_requests_saved', 0)}`) | "
            f"Opt candidate reqs: `{telem.get('optimization_candidate_requests', 0)}`"
        )
        st.caption(
            f"Early-stopped cases: `{telem.get('early_stopped_cases', 0)}` | "
            f"Saved via early stop: `{telem.get('requests_saved_early_stopping', 0)}` | "
            f"Total tokens: `{telem['total_tokens']}`"
        )

    is_busy = bool(st.session_state.get("active_operation"))

    with st.sidebar:
        st.divider()
        st.markdown(
            "<span class='forge-section-label'>Resume Saved Run</span>",
            unsafe_allow_html=True,
        )
        saved_runs = list_saved_runs(settings.artifacts_dir)
        if not saved_runs:
            st.caption("No saved run checkpoints found yet.")
        else:
            run_labels = {
                r["run_id"]: (
                    f"{r['run_id']} [{r['status']}] "
                    f"({r['completed_cases']}/{r['total_cases']} cases)"
                )
                for r in saved_runs
            }
            selected_run_id = st.selectbox(
                "Saved Run ID",
                options=[r["run_id"] for r in saved_runs],
                format_func=lambda rid: run_labels.get(rid, rid),
                key="selected_resume_run_id",
                disabled=is_busy,
            )
            selected_summary = next(r for r in saved_runs if r["run_id"] == selected_run_id)
            st.caption(
                f"Status: `{selected_summary['status']}` | "
                f"Completed: `{selected_summary['completed_cases']}` | "
                f"Remaining: `{selected_summary['remaining_cases']}`"
            )
            if st.button("Load Selected Run", type="secondary", disabled=is_busy):
                def _load_saved_run():
                    loaded = load_run_checkpoint(selected_summary["run_dir"])
                    brief = loaded.get("brief_input")
                    if brief is not None and brief.documents_dir.exists() and brief.documents_dir.is_dir():
                        docs, units = load_document_folder(
                            brief.documents_dir,
                            csv_rows_per_unit=loaded.get("csv_rows_per_unit", settings.csv_rows_per_unit),
                        )
                        loaded["documents"] = docs
                        loaded["document_units"] = units
                    else:
                        loaded.setdefault("document_units", [])
                    return loaded

                _execute_guarded_action(
                    "resume_load",
                    _load_saved_run,
                    section="phase4",
                    spinner_text="Loading saved checkpoint...",
                    rerun=True,
                )

    if is_busy:
        st.warning(
            f"Operation in progress: `{st.session_state['active_operation']}`. "
            "Action buttons are locked to prevent duplicate API calls."
        )

    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.markdown(
            """<div style="font-size:0.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
            color:var(--accent);margin:0.4rem 0 0.4rem;">Step 1 of 5</div>
            <h3 style="margin:0 0 1rem;font-size:1.25rem;font-weight:750;letter-spacing:-0.03em;color:var(--fg);">
            Business Brief &amp; Grounding Documents</h3>""",
            unsafe_allow_html=True,
        )
        brief_text = st.text_area(
            "Unseen Business Brief",
            placeholder="Describe the agent's purpose, audience, tone, must-answer topics, and must-refuse boundaries...",
            height=150,
            disabled=is_busy,
        )
        uploaded_files = st.file_uploader(
            "Upload Grounding Documents (PDF, DOCX, TXT, MD, CSV)",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            disabled=is_busy,
            key="grounding_file_uploader",
            help="Upload one or more domain documents for this run. Each new run uses an isolated corpus and vector collection.",
        )
        if uploaded_files:
            st.caption(f"Staged {len(uploaded_files)} file(s) for upload:")
            st.dataframe(
                [
                    {
                        "file_name": f.name,
                        "extension": Path(f.name).suffix.lower(),
                        "bytes": len(f.getvalue()),
                    }
                    for f in uploaded_files
                ],
                use_container_width=True,
            )

        with st.expander("Advanced: Use Existing Local Corpus Folder (Demos / CLI)", expanded=False):
            docs_dir_input = st.text_input(
                "Document Folder Path",
                value="",
                placeholder="e.g. demos/hr_policy/documents or demos/ship_spares/documents",
                disabled=is_busy,
                help="Optional local folder path when testing built-in demo corpora instead of uploading files.",
            )
            if st.button("Inspect Document Folder", type="secondary", disabled=is_busy):
                st.session_state["last_error"] = None
                target_dir = Path(docs_dir_input.strip())
                if not docs_dir_input.strip() or not target_dir.exists() or not target_dir.is_dir():
                    st.error(f"Folder not found: `{docs_dir_input or '(empty)'}`")
                else:
                    docs, units = load_document_folder(
                        target_dir, csv_rows_per_unit=settings.csv_rows_per_unit
                    )
                    if not docs:
                        st.info("No supported documents (.pdf, .docx, .csv, .txt, .md) found in that folder.")
                    else:
                        st.success(f"Found {len(docs)} document(s) across {len(units)} source unit(s).")
                        st.dataframe(
                            [
                                {
                                    "doc_id": d.doc_id,
                                    "file_name": d.file_name,
                                    "file_type": d.file_type,
                                    "units": d.unit_count,
                                    "bytes": d.size_bytes,
                                }
                                for d in docs
                            ],
                            use_container_width=True,
                        )

        existing_state = st.session_state.get("factory_state")
        has_existing_intake = bool(
            existing_state
            and (
                existing_state.get("clarifying_questions")
                or existing_state.get("requirements")
            )
        )
        confirm_intake_rerun = False
        if has_existing_intake:
            upload_meta_active = st.session_state.get("uploaded_corpus_meta") or {}
            if uploaded_files:
                staged_sig = sorted((f.name, len(f.getvalue())) for f in uploaded_files)
                active_sig = upload_meta_active.get("upload_signature")
                if staged_sig != active_sig:
                    st.info(
                        "New uploaded file(s) staged. Check confirmation below and click "
                        "'Start Intake Interview' to build a fresh isolated run with these files."
                    )
            confirm_intake_rerun = st.checkbox(
                "Confirm starting a new Intake Interview (replaces current session state)",
                value=False,
                key="confirm_intake_rerun",
                disabled=is_busy,
            )

        if st.button(
            "Start Intake Interview",
            type="primary",
            disabled=is_busy or not ready or (has_existing_intake and not confirm_intake_rerun),
        ):
            def _run_intake():
                if uploaded_files:
                    upload_res = prepare_uploaded_corpus(
                        uploaded_files,
                        artifacts_dir=settings.artifacts_dir,
                        csv_rows_per_unit=settings.csv_rows_per_unit,
                    )
                    new_upload_meta = {
                        "run_id": upload_res.run_id,
                        "corpus_sha256": upload_res.corpus_sha256,
                        "saved_files": upload_res.saved_files,
                        "renamed_files": upload_res.renamed_files,
                        "documents_dir": str(upload_res.documents_dir),
                        "upload_signature": sorted((f.name, len(f.getvalue())) for f in uploaded_files),
                    }
                    brief = BriefInput(
                        brief_text=brief_text,
                        documents_dir=upload_res.documents_dir,
                        run_id=upload_res.run_id,
                    )
                    run_id = upload_res.run_id
                elif docs_dir_input.strip():
                    new_upload_meta = None
                    run_id = f"run_{uuid4().hex[:8]}"
                    brief = BriefInput(
                        brief_text=brief_text,
                        documents_dir=Path(docs_dir_input.strip()),
                        run_id=run_id,
                    )
                else:
                    raise UploadValidationError(
                        "Please upload one or more documents (PDF, DOCX, TXT, MD, CSV) "
                        "or provide an existing Document Folder Path under Advanced."
                    )

                # Clear previous chat/upload state only after validation and staging succeed.
                st.session_state["chat_history"] = []
                st.session_state["chat_corpus_key"] = None
                st.session_state["uploaded_corpus_meta"] = new_upload_meta

                graph = build_factory_graph(settings=settings)
                return graph.invoke(
                    {
                        "run_id": run_id,
                        "brief_input": brief,
                        "csv_rows_per_unit": settings.csv_rows_per_unit,
                        "stop_after_phase2": True,
                    }
                )

            _execute_guarded_action(
                "intake_interview",
                _run_intake,
                section="phase1",
                spinner_text="Validating documents & generating clarifying questions...",
                rerun=False,
            )

        upload_meta = st.session_state.get("uploaded_corpus_meta")
        active_state_for_meta = st.session_state.get("factory_state") or {}
        if upload_meta and active_state_for_meta.get("run_id") == upload_meta.get("run_id"):
            st.success(
                f"Isolated upload corpus ready for run `{upload_meta['run_id']}`: "
                f"{len(upload_meta['saved_files'])} document(s) "
                f"(SHA-256: `{upload_meta['corpus_sha256'][:12]}`)."
            )
            if upload_meta.get("renamed_files"):
                st.info(
                    "Duplicate filenames were safely renamed: "
                    + ", ".join(f"{orig} -> {saved}" for orig, saved in upload_meta["renamed_files"].items())
                )

        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])

        state = st.session_state.get("factory_state")
        if state and state.get("clarifying_questions"):
            st.subheader("2. Clarifying Questions (Max 5)")
            questions = state["clarifying_questions"]
            answers_map: dict[str, str] = {}
            with st.form("clarifying_answers_form"):
                for idx, q in enumerate(questions, start=1):
                    st.markdown(f"**Q{idx} (`{q.target_field}`):** {q.question}")
                    st.caption(f"Why asked: {q.rationale}")
                    answers_map[q.question_id] = st.text_input(
                        f"Answer for Q{idx} (leave blank to skip / partial answers allowed)",
                        key=f"ans_{q.question_id}",
                    )
                approve_gaps = st.checkbox(
                    "Allow proceeding if unsupported topics remain (requester override)",
                    value=False,
                )
                submitted = st.form_submit_button(
                    "Synthesize Requirements & Run Groundability Guard",
                    disabled=is_busy,
                )

            if submitted:
                def _run_phase2():
                    graph = build_factory_graph(settings=settings)
                    return graph.invoke(
                        {
                            **state,
                            "clarifying_answers": answers_map,
                            "interview_completed": True,
                            "approved_with_gaps": approve_gaps,
                            "stop_after_phase2": True,
                        }
                    )

                _execute_guarded_action(
                    "groundability_guard",
                    _run_phase2,
                    section="phase2",
                    spinner_text="Synthesizing requirements & checking corpus groundability...",
                    rerun=True,
                )

    with col_right:
        st.markdown(
            """<div style="font-size:0.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
            color:var(--accent);margin:0.4rem 0 0.4rem;">Steps 2–5</div>
            <h3 style="margin:0 0 1rem;font-size:1.25rem;font-weight:750;letter-spacing:-0.03em;color:var(--fg);">
            Requirements, Agent &amp; Evaluation Output</h3>""",
            unsafe_allow_html=True,
        )
        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])
        state = st.session_state.get("factory_state")

        if not state:
            st.info(
                "Run the Intake Interview to generate requirements, groundability findings, "
                "a declarative agent, and automated evaluations."
            )
            return

        docs = state.get("documents") or []
        st.caption(f"Ingested documents: {len(docs)} | Units: {len(state.get('document_units') or [])}")

        req = state.get("requirements")
        if req is not None:
            st.markdown("#### Structured Requirements")
            st.write(f"**Domain:** {req.domain_name}")
            if req.purpose:
                st.write(f"**Purpose:** {req.purpose}")
            st.write(f"**Scope:** {req.scope}")
            st.write(f"**Audience:** {req.audience} | **Tone:** {req.tone}")
            if req.expected_response_behavior:
                st.write(f"**Response Behaviour:** {req.expected_response_behavior}")
            st.write("**Must-Answer Topics:**", req.must_answer_topics)
            st.write("**Must-Refuse Topics:**", req.must_refuse_topics)
            if req.constraints:
                st.write("**Constraints:**", req.constraints)
            if req.assumptions:
                st.write("**Assumptions:**", req.assumptions)

        report = state.get("groundability")
        if report is not None:
            st.markdown("#### Groundability Guard Report")
            verdict_label = report.overall_verdict.value
            if report.can_proceed and not report.ungrounded_topics:
                st.success(f"Overall Verdict: `{verdict_label}` — Approved for Agent Generation.")
            elif report.can_proceed:
                st.warning(
                    f"Overall Verdict: `{verdict_label}` — Approved with gaps: {report.ungrounded_topics}"
                )
            else:
                st.error(
                    f"Overall Verdict: `{verdict_label}` — Blocked before Agent Generation. "
                    f"Unsupported topics: {report.ungrounded_topics}"
                )

            for finding in report.findings:
                badge = {
                    GroundabilityVerdict.SUPPORTED: "SUPPORTED",
                    GroundabilityVerdict.PARTIALLY_SUPPORTED: "PARTIAL",
                    GroundabilityVerdict.UNSUPPORTED: "UNSUPPORTED",
                }[finding.verdict]
                scope_tag = " [OUT-OF-SCOPE]" if finding.is_out_of_scope else ""
                with st.expander(f"[{badge}]{scope_tag} {finding.topic}"):
                    st.write(f"**Evidence Summary:** {finding.evidence_summary}")
                    st.write(f"**Supporting Sources:** {finding.supporting_sources or ['None']}")
                    if finding.gap_description:
                        st.write(f"**Missing / Gap:** {finding.gap_description}")

            if report.recommendations:
                st.markdown("**Recommendations:**")
                for rec in report.recommendations:
                    st.markdown(f"- {rec}")

            st.markdown("#### Declarative Agent Generator (Phase 3)")
            can_generate = ready and report.can_proceed
            agent_exists = state.get("agent_definition") is not None
            confirm_agent_regen = False
            if agent_exists:
                confirm_agent_regen = st.checkbox(
                    "Confirm regenerating declarative agent",
                    value=False,
                    key="confirm_agent_regen",
                    disabled=is_busy,
                )
            if st.button(
                "Generate Agent",
                type="primary",
                disabled=is_busy or not can_generate or (agent_exists and not confirm_agent_regen),
            ):
                def _run_phase3():
                    graph = build_factory_graph(settings=settings)
                    next_input = {
                        **state,
                        "stop_after_phase2": False,
                        "stop_after_phase3": True,
                    }
                    if confirm_agent_regen:
                        next_input.pop("agent_definition", None)
                        next_input.pop("agent_artifact_path", None)
                    return graph.invoke(next_input)

                _execute_guarded_action(
                    "generate_agent",
                    _run_phase3,
                    section="phase3",
                    spinner_text="Generating declarative agent configuration...",
                    rerun=True,
                )
            if st.session_state["last_error"] and st.session_state["last_error_section"] == "phase3":
                st.error(st.session_state["last_error"])

        agent = state.get("agent_definition")
        if agent is not None:
            artifact_path = state.get("agent_artifact_path")
            st.success(f"Generated Agent: `{agent.name}` (`{agent.agent_id}`, v{agent.version})")
            if artifact_path:
                st.caption(f"Persisted to: `{artifact_path}`")

            with st.expander("System Prompt"):
                st.text(agent.system_prompt)
            with st.expander("Retrieval & Chunking Settings"):
                st.json({"chunking": agent.chunking.model_dump(), "retrieval": agent.retrieval.model_dump()})
            with st.expander("Selected Tools & Reasons"):
                for tool in agent.tools:
                    reason = tool.config.get("reason", tool.purpose)
                    st.markdown(f"- **`{tool.name}`**: {tool.purpose} *(Reason: {reason})*")
            with st.expander("Output Contract"):
                st.json(agent.output_contract.model_dump())

            agent_json = agent.model_dump_json(indent=2)
            st.download_button(
                "Export Agent JSON",
                data=agent_json,
                file_name=f"{agent.agent_id}_v{agent.version}.json",
                mime="application/json",
            )

            st.markdown("#### Evaluation Generator & Hybrid Judge (Phase 4)")
            eval_cases = state.get("eval_cases") or []
            eval_set_ready = len(eval_cases) >= settings.min_eval_cases
            eval_completed = is_evaluation_already_completed(state, agent)
            cp_done = state.get("checkpoint_completed_cases")
            cp_total = state.get("checkpoint_total_cases")
            if (
                eval_set_ready
                and not eval_completed
                and cp_done is not None
                and cp_total is not None
                and cp_total > 0
            ):
                cp_rem = max(0, cp_total - cp_done)
                st.info(
                    f"Checkpoint progress: **{cp_done}/{cp_total}** cases completed "
                    f"(**{cp_rem}** remaining). Running evaluations will skip completed cases."
                )

            confirm_eval_gen_rerun = False
            if eval_set_ready:
                confirm_eval_gen_rerun = st.checkbox(
                    "Confirm regenerating evaluation set",
                    value=False,
                    key="confirm_eval_gen_rerun",
                    disabled=is_busy,
                )

            confirm_eval_rerun = False
            if eval_completed:
                suite_short = (state.get("eval_suite_sha256") or "")[:12]
                st.info(
                    f"Evaluation already completed for `{agent.agent_id}` v{agent.version} "
                    f"(suite SHA-256: `{suite_short}`). Confirm below to intentionally rerun."
                )
                confirm_eval_rerun = st.checkbox(
                    "Confirm rerun of completed evaluations",
                    value=False,
                    key="confirm_eval_rerun",
                    disabled=is_busy,
                )

            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                if st.button(
                    "Generate Evaluation Set",
                    type="secondary",
                    disabled=is_busy or not ready or (eval_set_ready and not confirm_eval_gen_rerun),
                ):
                    def _run_eval_gen():
                        graph = build_factory_graph(settings=settings)
                        next_input = {
                            **state,
                            "stop_after_phase2": False,
                            "stop_after_phase3": False,
                            "stop_after_eval_generation": True,
                            "stop_after_phase4": True,
                        }
                        if confirm_eval_gen_rerun:
                            next_input.pop("eval_cases", None)
                            next_input.pop("eval_suite_sha256", None)
                            next_input.pop("evaluated_signature", None)
                        return graph.invoke(next_input)

                    _execute_guarded_action(
                        "generate_evals",
                        _run_eval_gen,
                        section="phase4",
                        spinner_text="Generating 20+ evaluation cases...",
                        rerun=True,
                    )

            with eval_col2:
                if st.button(
                    "Run Evaluations & Judge",
                    type="primary",
                    disabled=(
                        is_busy
                        or not ready
                        or not eval_set_ready
                        or (eval_completed and not confirm_eval_rerun)
                    ),
                ):
                    def _run_eval_judge():
                        graph = build_factory_graph(settings=settings)
                        return graph.invoke(
                            {
                                **state,
                                "stop_after_phase2": False,
                                "stop_after_phase3": False,
                                "stop_after_eval_generation": False,
                                "stop_after_phase4": True,
                            }
                        )

                    _execute_guarded_action(
                        "run_evaluations",
                        _run_eval_judge,
                        section="phase4",
                        spinner_text=(
                            "Running 20 evaluation cases & Hybrid Judge "
                            "(automatically pausing for Free Tier 15-RPM limits, ~3–5 min)..."
                        ),
                        mark_evaluated=True,
                        rerun=True,
                    )

            if st.session_state["last_error"] and st.session_state["last_error_section"] == "phase4":
                st.error(st.session_state["last_error"])

            if eval_cases:
                st.info(f"Generated Evaluation Cases: **{len(eval_cases)}**")
                grouped = defaultdict(list)
                for case in eval_cases:
                    grouped[case.category.value].append(case)
                for category_name, items in grouped.items():
                    with st.expander(f"Category `{category_name}` ({len(items)} cases)"):
                        for item in items:
                            st.markdown(f"**{item.case_id}:** {item.question}")
                            st.caption(
                                f"Behavior: `{item.expected_behavior.value}` | "
                                f"Sources: {item.source_documents or ['None']} | "
                                f"Expected Facts: {item.expected_facts or ['Refuse']}"
                            )
                cases_json = json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2)
                st.download_button(
                    "Export Evaluation Cases JSON",
                    data=cases_json,
                    file_name=f"{agent.agent_id}_eval_cases.json",
                    mime="application/json",
                )

            eval_results = state.get("eval_results") or []
            best_results = state.get("best_results") or eval_results
            best_sc = state.get("best_scorecard")
            best_agent = state.get("best_agent") or agent
            scorecards = state.get("scorecards") or []
            suite_hash = (state.get("eval_suite_sha256") or "")[:12] or "n/a"

            if best_sc is None and scorecards:
                complete_scs = [sc for sc in scorecards if not getattr(sc, "is_partial", False)]
                best_sc = max(complete_scs, key=lambda sc: sc.pass_rate) if complete_scs else scorecards[-1]

            if best_sc is not None:
                st.markdown("#### Evaluation Scorecard — Retained Best Complete")
                st.caption(
                    f"Retained Best Agent: `{best_agent.agent_id}` v{best_agent.version} | "
                    f"Iteration {best_sc.iteration} | Evaluated: {best_sc.passed_cases}/{best_sc.total_cases} cases | "
                    f"Suite SHA-256: `{suite_hash}`"
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pass Rate", f"{best_sc.pass_rate:.0%}")
                m2.metric("Passed / Total", f"{best_sc.passed_cases}/{best_sc.total_cases}")
                m3.metric("Grounding Rate", f"{best_sc.grounding_rate:.0%}")
                m4.metric("Citation Accuracy", f"{best_sc.citation_accuracy:.0%}")

                pdf_bytes = build_evaluation_report_pdf_from_state(
                    {
                        **state,
                        "best_scorecard": best_sc,
                        "best_agent": best_agent,
                        "best_results": best_results or eval_results,
                    }
                )
                st.download_button(
                    "Download Evaluation Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{best_agent.agent_id}_v{best_agent.version}_evaluation_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    key="download_eval_report_pdf",
                )
                st.json(best_sc.model_dump(mode="json"))

            latest_sc = scorecards[-1] if scorecards else None
            latest_cand_results = state.get("latest_candidate_results")
            latest_cand_agent = state.get("latest_candidate_agent")
            opt_logs_all = state.get("optimization_logs") or []
            show_latest_candidate = latest_sc is not None and (
                best_sc is None
                or latest_sc.iteration != best_sc.iteration
                or getattr(latest_sc, "is_partial", False)
            )
            if show_latest_candidate and latest_sc is not None:
                cand_ver = (
                    latest_cand_agent.version
                    if latest_cand_agent is not None
                    else (opt_logs_all[-1].agent_version_before + 1 if opt_logs_all else agent.version)
                )
                status_note = "Rejected / Partial" if getattr(latest_sc, "is_partial", False) else "Latest Candidate"
                st.markdown(f"#### Latest Candidate Scorecard ({status_note})")
                if getattr(latest_sc, "is_partial", False):
                    expected_total = latest_sc.expected_total_cases or latest_sc.total_cases
                    st.warning(
                        f"EARLY-STOPPED / PARTIAL EVALUATION: evaluated {latest_sc.total_cases}/"
                        f"{expected_total} cases (not directly comparable to full-suite scores)."
                    )
                st.caption(
                    f"Candidate Agent: `{agent.agent_id}` v{cand_ver} | "
                    f"Iteration {latest_sc.iteration} | Evaluated: {latest_sc.passed_cases}/{latest_sc.total_cases} cases | "
                    f"Suite SHA-256: `{suite_hash}`"
                )
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("Candidate Pass Rate", f"{latest_sc.pass_rate:.0%}")
                total_label = "Passed / Evaluated" if getattr(latest_sc, "is_partial", False) else "Passed / Total"
                cm2.metric(total_label, f"{latest_sc.passed_cases}/{latest_sc.total_cases}")
                cm3.metric("Grounding Rate", f"{latest_sc.grounding_rate:.0%}")
                cm4.metric("Citation Accuracy", f"{latest_sc.citation_accuracy:.0%}")
                if latest_cand_results:
                    cand_failures = [r for r in latest_cand_results if not r.passed]
                    with st.expander(
                        f"Candidate v{cand_ver} Failures ({len(cand_failures)} / {len(latest_cand_results)} evaluated)",
                        expanded=False,
                    ):
                        for fail in cand_failures:
                            st.markdown(f"- `[FAIL: {fail.failure_classification.value}] {fail.case_id}: {fail.question}`")

            displayed_results = best_results or eval_results
            if displayed_results:
                res_iter = best_sc.iteration if best_sc is not None else state.get("iteration", 0)
                failures = [r for r in displayed_results if not r.passed]
                if failures:
                    st.error(
                        f"Retained Best Failures (v{best_agent.version}, Iter {res_iter}, Suite `{suite_hash}`): "
                        f"{len(failures)} / {len(displayed_results)}"
                    )
                    for fail in failures:
                        with st.expander(
                            f"[FAIL: {fail.failure_classification.value}] {fail.case_id}: {fail.question}"
                        ):
                            st.write(f"**Agent Answer:** {fail.agent_answer}")
                            st.write(f"**Judge Reasoning:** {fail.judge_reasoning}")
                            if fail.deterministic_issues:
                                st.write(f"**Deterministic Issues:** {fail.deterministic_issues}")
                            if fail.disagreements:
                                st.write(f"**Disagreements:** {fail.disagreements}")
                else:
                    st.success("All retained-best evaluation cases passed!")

                with st.expander(f"All Individual Judge Results ({len(displayed_results)})"):
                    for res in displayed_results:
                        status_tag = "PASS" if res.passed else f"FAIL ({res.failure_classification.value})"
                        st.markdown(f"- **`{res.case_id}` [{status_tag}]** — {res.question}")
                        st.caption(
                            f"Grounding: {res.grounding_score:.2f} | Correctness: {res.correctness_score:.2f} | "
                            f"Cited: {res.cited_sources or ['None']}"
                        )

                results_json = json.dumps([r.model_dump(mode="json") for r in displayed_results], indent=2)
                st.download_button(
                    "Export Evaluation Results JSON",
                    data=results_json,
                    file_name=f"{agent.agent_id}_eval_results.json",
                    mime="application/json",
                )

            st.markdown("#### Critic–Optimizer & Honest Final Report (Phase 5)")
            opt_col1, opt_col2 = st.columns([1, 1])
            with opt_col1:
                max_iters_ui = st.number_input(
                    "Optimization Budget (0–3 iterations)",
                    min_value=0,
                    max_value=3,
                    value=settings.max_optimization_iterations,
                    step=1,
                )
            with opt_col2:
                used_iters = len(state.get("optimization_logs") or [])
                st.metric("Remaining Budget", f"{max(0, int(max_iters_ui) - used_iters)} / {int(max_iters_ui)}")

            opt_completed = is_optimization_already_completed(state, agent)
            confirm_opt_rerun = False
            if opt_completed:
                st.info("Optimization has already completed for this agent and evaluation suite.")
                confirm_opt_rerun = st.checkbox(
                    "Confirm rerun of optimization",
                    value=False,
                    key="confirm_opt_rerun",
                    disabled=is_busy,
                )

            if st.button(
                "Optimize Agent",
                type="primary",
                disabled=(
                    is_busy
                    or not ready
                    or len(eval_cases) < settings.min_eval_cases
                    or not eval_results
                    or (opt_completed and not confirm_opt_rerun)
                ),
            ):
                def _run_phase5():
                    graph = build_factory_graph(settings=settings)
                    return graph.invoke(
                        {
                            **state,
                            "max_iterations": int(max_iters_ui),
                            "stop_after_phase2": False,
                            "stop_after_phase3": False,
                            "stop_after_eval_generation": False,
                            "stop_after_phase4": False,
                        }
                    )

                _execute_guarded_action(
                    "optimize_agent",
                    _run_phase5,
                    section="phase5",
                    spinner_text=(
                        "Running Critic–Optimizer iterations & re-evaluations "
                        "(~3–5 min per iteration on Free Tier)..."
                    ),
                    mark_optimized=True,
                    rerun=True,
                )

            if st.session_state["last_error"] and st.session_state["last_error_section"] == "phase5":
                st.error(st.session_state["last_error"])

            opt_logs = state.get("optimization_logs") or []
            if opt_logs:
                st.markdown("##### Per-Iteration Critic–Optimizer History")
                for log in opt_logs:
                    if log.early_stopped:
                        badge = "EARLY-STOPPED / PARTIAL (REJECTED)"
                    elif log.accepted:
                        badge = "ACCEPTED"
                    else:
                        badge = "REJECTED"
                    with st.expander(
                        f"Iteration {log.iteration} [{badge}] — Categories: {[c.value for c in log.observed_failure_categories]}"
                    ):
                        st.write(f"**Diagnosis:** {log.diagnosis_summary}")
                        st.write(f"**Failed Case IDs:** {log.failed_case_ids}")
                        st.write("**Proposed Changes:**", [c.model_dump() for c in log.proposed_changes])
                        st.write("**Applied Changes:**", [c.model_dump() for c in log.targeted_changes])
                        if log.rejected_changes:
                            st.write("**Rejected Changes:**", [r.model_dump() for r in log.rejected_changes])
                        if log.after_scorecard is not None:
                            if log.early_stopped or getattr(log.after_scorecard, "is_partial", False):
                                evaluated_cnt = log.evaluated_cases_count or log.after_scorecard.total_cases
                                expected_cnt = log.after_scorecard.expected_total_cases or log.before_scorecard.total_cases
                                st.write(
                                    f"**EARLY-STOPPED / PARTIAL Evaluation:** {evaluated_cnt}/{expected_cnt} "
                                    f"cases evaluated ({log.early_stopped_cases} skipped, {log.requests_saved} "
                                    f"requests saved). Partial metrics are not compared against the full-suite baseline."
                                )
                            else:
                                st.write(
                                    f"**Scorecard:** {log.before_scorecard.pass_rate:.0%} -> "
                                    f"{log.after_scorecard.pass_rate:.0%} | **Deltas:** {log.metric_deltas}"
                                )
                        st.write(f"**Decision:** {log.acceptance_reason}")
                        st.write(f"**Remaining Failures:** {log.remaining_failed_case_ids}")

            final_report = state.get("final_report")
            if final_report is not None:
                st.markdown("##### Honest Final Report")
                if final_report.passing_criteria_met:
                    st.success(
                        f"Passing criteria met ({final_report.final_pass_rate:.0%} >= "
                        f"{final_report.target_pass_rate:.0%}). Stop reason: `{final_report.stop_reason}`"
                    )
                else:
                    st.warning(
                        f"Passing criteria NOT met ({final_report.final_pass_rate:.0%} < "
                        f"{final_report.target_pass_rate:.0%}) after {final_report.iterations_used} iteration(s). "
                        f"Stop reason: `{final_report.stop_reason}`"
                    )

                b_sc = final_report.baseline_scorecard
                f_sc = final_report.final_scorecard
                if b_sc and f_sc:
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric(
                        "Final Pass Rate",
                        f"{f_sc.pass_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('pass_rate', 0.0):+.0%}",
                    )
                    rm2.metric(
                        "Grounding Rate",
                        f"{f_sc.grounding_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('grounding_rate', 0.0):+.0%}",
                    )
                    rm3.metric(
                        "Citation Accuracy",
                        f"{f_sc.citation_accuracy:.0%}",
                        delta=f"{final_report.metric_deltas.get('citation_accuracy', 0.0):+.0%}",
                    )
                    rm4.metric(
                        "Contract Compliance",
                        f"{f_sc.contract_compliance_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('contract_compliance_rate', 0.0):+.0%}",
                    )

                if final_report.honest_limitations:
                    with st.expander("Honest Limitations & Remaining Failures", expanded=True):
                        for item in final_report.honest_limitations:
                            st.markdown(f"- {item}")

                rep_col1, rep_col2 = st.columns(2)
                with rep_col1:
                    st.download_button(
                        "Export Final Report JSON",
                        data=final_report.model_dump_json(indent=2),
                        file_name=f"{final_report.run_id}_final_report.json",
                        mime="application/json",
                    )
                with rep_col2:
                    st.download_button(
                        "Export Final Agent JSON",
                        data=final_report.final_agent.model_dump_json(indent=2),
                        file_name=f"{final_report.final_agent.agent_id}_final_v{final_report.final_agent.version}.json",
                        mime="application/json",
                    )

            active_agent = state.get("best_agent") or state.get("agent_definition")
            active_brief = state.get("brief_input")
            active_docs_dir = active_brief.documents_dir if active_brief is not None else None
            active_units = state.get("document_units") or []
            if not active_units and active_docs_dir is not None and active_docs_dir.exists():
                chat_csv_rows = (
                    active_agent.chunking.csv_rows_per_unit
                    if active_agent is not None
                    else state.get("csv_rows_per_unit", settings.csv_rows_per_unit)
                )
                _, active_units = load_document_folder(
                    active_docs_dir,
                    csv_rows_per_unit=chat_csv_rows,
                )

            if active_agent is not None and active_units:
                st.divider()
                st.markdown("#### 6. Test Generated Agent (Generic Chat Runner)")
                corpus_key = (
                    state.get("run_id"),
                    active_agent.agent_id,
                    active_agent.version,
                    active_agent.retrieval.collection_name,
                    str(active_docs_dir) if active_docs_dir else "",
                )
                if st.session_state.get("chat_corpus_key") != corpus_key:
                    st.session_state["chat_history"] = []
                    st.session_state["chat_corpus_key"] = corpus_key

                st.caption(
                    f"Using `{active_agent.name}` v{active_agent.version} | "
                    f"Isolated collection: `{active_agent.retrieval.collection_name}` | "
                    f"Sources: {', '.join(active_agent.source_documents) or 'None'}"
                )
                with st.form("generic_agent_chat_form", clear_on_submit=True):
                    chat_question = st.text_input(
                        "Ask a question grounded in this run's documents",
                        key="chat_runner_question",
                        disabled=is_busy,
                    )
                    ask_submitted = st.form_submit_button(
                        "Ask Generated Agent",
                        disabled=is_busy or not ready,
                    )

                if ask_submitted and chat_question.strip():
                    def _run_chat_query():
                        run_store = build_isolated_vector_store(
                            active_agent,
                            active_units,
                            persist_dir=settings.chroma_persist_dir,
                            collection_name=active_agent.retrieval.collection_name,
                            reset=True,
                        )
                        run_registry = build_default_registry(
                            vector_store=run_store,
                            documents_dir=active_docs_dir,
                        )
                        runner = DeclarativeAgentRunner(
                            active_agent,
                            vector_store=run_store,
                            gemini_client=GeminiClient(settings=settings),
                            tool_registry=run_registry,
                            documents_dir=active_docs_dir,
                        )
                        resp = runner.run(chat_question.strip())
                        history = list(st.session_state.get("chat_history") or [])
                        history.append(
                            {
                                "question": chat_question.strip(),
                                "answer": resp.answer,
                                "refused": resp.refused,
                                "citations": resp.citations,
                                "tools_used": resp.tools_used,
                                "evidence": [ev.model_dump() for ev in resp.evidence],
                            }
                        )
                        st.session_state["chat_history"] = history
                        return None

                    _execute_guarded_action(
                        "chat_runner",
                        _run_chat_query,
                        section="chat_runner",
                        spinner_text="Querying isolated agent collection...",
                        update_factory_state=False,
                        rerun=True,
                    )

                if st.session_state["last_error"] and st.session_state["last_error_section"] == "chat_runner":
                    st.error(st.session_state["last_error"])

                for turn in reversed(st.session_state.get("chat_history") or []):
                    with st.expander(f"Q: {turn['question']}", expanded=True):
                        if turn["refused"]:
                            st.warning("Agent refused request (out-of-scope or unsupported by corpus).")
                        st.markdown(turn["answer"])
                        st.caption(
                            f"Citations: {turn['citations'] or ['None']} | "
                            f"Tools: {turn['tools_used'] or ['None']}"
                        )


if __name__ == "__main__":
    main()
