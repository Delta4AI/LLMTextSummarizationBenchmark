#!/usr/bin/env python3
"""Human evaluation server for the text summarization benchmark.

Serves a two-step web interface where reviewers rate model-generated
summaries on SummEval quality dimensions. Each reviewer gets a unique
token URL with progress saved to a per-reviewer JSON file.

Usage
-----
    python human_evaluation_server.py \\
        openai_gpt-4o anthropic_claude-opus-4-20250514 \\
        local:textrank ollama_gemma3:270M \\
        --port 9987
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import logging
import os
import random
import re
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── Paths ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS = (
    ROOT
    / "Output"
    / "llm_summarization_benchmark"
    / "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
    / "detailed_scores_per_paper.json"
)
DEFAULT_GOLDSTANDARD = (
    ROOT / "Resources" / "text_summarization_goldstandard_data.json"
)
DEFAULT_DATA_DIR = ROOT / "Output" / "scripts" / "human_evaluation_data"
DEFAULT_PORT = 9987
DEFAULT_NUM_PAPERS = 20
MAX_NAME_LENGTH = 200
MAX_COMMENT_LENGTH = 2000
MAX_BODY_BYTES = 64 * 1024  # 64 KB

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rating dimensions ────────────────────────────────────────────────────

# SummEval dimensions (Fabbri et al., 2021, TACL).
# https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373
DETAILED_CRITERIA = [
    {
        "key": "coherence",
        "label": "Coherence",
        "description": "Is the summary well-structured and logically organized?",
        "anchors": {1: "Confusing, disorganized", 3: "Somewhat organized", 5: "Clear logical flow"},
    },
    {
        "key": "consistency",
        "label": "Consistency",
        "description": "Is the summary factually consistent with the source? No hallucinations?",
        "anchors": {1: "Major factual errors", 3: "Minor inaccuracies", 5: "Fully consistent"},
    },
    {
        "key": "fluency",
        "label": "Fluency",
        "description": "Is the summary grammatical, well-written, and readable?",
        "anchors": {1: "Difficult to understand", 3: "Mostly readable", 5: "Clear and well-written"},
    },
    {
        "key": "relevance",
        "label": "Relevance",
        "description": "Does the summary capture the important information from the source?",
        "anchors": {1: "Missing most key points", 3: "Captures some key points", 5: "Captures all key points"},
    },
]

# ── Thread safety ────────────────────────────────────────────────────────

_write_lock = threading.Lock()
_TOKEN_RE = re.compile(r"^r_[0-9a-f]{12}$")

# ── Data loading ─────────────────────────────────────────────────────────


def load_evaluation_data(
    results_path: Path,
    goldstandard_path: Path,
    model_names: list[str],
    num_papers: int,
) -> dict:
    """Load gold standard + results, select papers for evaluation."""
    log.info("Loading gold-standard data from %s", goldstandard_path)
    with open(goldstandard_path, encoding="utf-8") as f:
        goldstandard = json.load(f)

    # Build journal lookup: paper_id -> section_type
    paper_journal: dict[str, str] = {}
    journal_papers: dict[str, list[str]] = {}
    paper_meta: dict[str, dict] = {}
    for paper in goldstandard:
        pid = paper["id"]
        journal = paper.get("section_type", "Unknown")
        paper_journal[pid] = journal
        journal_papers.setdefault(journal, []).append(pid)
        paper_meta[pid] = paper

    log.info(
        "Loaded %d papers across %d journal categories",
        len(goldstandard),
        len(journal_papers),
    )

    log.info("Loading results from %s (this may take a moment)...", results_path)
    with open(results_path, encoding="utf-8") as f:
        all_results = json.load(f)

    # Validate model names
    missing = [m for m in model_names if m not in all_results]
    if missing:
        log.error("Model(s) not found in results: %s", ", ".join(missing))
        log.info("Available models:\n  %s", "\n  ".join(sorted(all_results.keys())))
        sys.exit(1)

    # Extract only the 4 models' data, indexed by paper id
    model_results: dict[str, dict[str, dict]] = {}
    for model in model_names:
        model_results[model] = {p["id"]: p for p in all_results[model]}
    del all_results  # free ~240 MB

    # Select papers: one per top-N journal, must have valid summaries
    journal_counts = Counter(paper_journal.values())
    top_journals = [j for j, _ in journal_counts.most_common()]

    selected_papers: list[dict] = []
    for journal in top_journals:
        if len(selected_papers) >= num_papers:
            break
        candidates = sorted(journal_papers[journal])
        for pid in candidates:
            has_all = all(
                pid in model_results[m]
                and model_results[m][pid].get("extracted_response")
                for m in model_names
            )
            if has_all:
                gs = paper_meta[pid]
                # Split highlights back into individual bullet points
                # using the <br>-separated summaries_decodable field
                decodable = gs.get("summaries_decodable", [""])[0]
                highlights = (
                    [h.strip() for h in decodable.split("<br>") if h.strip()]
                    if decodable
                    else gs["summaries"]
                )
                selected_papers.append(
                    {
                        "id": pid,
                        "title": gs["title"],
                        "abstract": gs["abstract"],
                        "reference_highlights": highlights,
                        "journal": journal,
                        "summaries": {
                            m: model_results[m][pid]["extracted_response"]
                            for m in model_names
                        },
                    }
                )
                break

    if len(selected_papers) < num_papers:
        log.warning(
            "Only found %d valid papers (requested %d)",
            len(selected_papers),
            num_papers,
        )

    total = len(selected_papers) * len(model_names)
    n_journals = len({p["journal"] for p in selected_papers})
    log.info(
        "Selected %d papers across %d journals → %d assessments per reviewer",
        len(selected_papers),
        n_journals,
        total,
    )

    return {
        "papers": selected_papers,
        "models": model_names,
        "total_assessments": total,
    }


# ── Reviewer management ─────────────────────────────────────────────────


def generate_token() -> str:
    return f"r_{os.urandom(6).hex()}"


def _is_valid_token(token: str) -> bool:
    return bool(_TOKEN_RE.match(token))


def _reviewer_path(data_dir: Path, token: str) -> Path:
    safe = "".join(c for c in token if c.isalnum() or c == "_")
    return data_dir / f"reviewer_{safe}.json"


def load_reviewer(data_dir: Path, token: str) -> dict | None:
    if not token or not _is_valid_token(token):
        return None
    path = _reviewer_path(data_dir, token)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_reviewer(data_dir: Path, reviewer: dict) -> None:
    """Atomic write to disk. Caller MUST hold _write_lock."""
    path = _reviewer_path(data_dir, reviewer["token"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reviewer, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_reviewer(data_dir: Path, reviewer: dict) -> None:
    with _write_lock:
        _write_reviewer(data_dir, reviewer)


def create_reviewer(
    data_dir: Path,
    name: str,
    eval_data: dict,
    rating_mode: str,
) -> dict:
    token = generate_token()
    rng = random.Random(token)

    assignments = [
        {"paper_index": pi, "model": model}
        for pi in range(len(eval_data["papers"]))
        for model in eval_data["models"]
    ]
    rng.shuffle(assignments)

    reviewer = {
        "token": token,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rating_mode": rating_mode,
        "assignments": assignments,
        "assessments": [],
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    save_reviewer(data_dir, reviewer)
    log.info("Created reviewer '%s' with token %s", name, token)
    return reviewer


# ── HTML templates ───────────────────────────────────────────────────────

LANDING_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Evaluation – Text Summarization Benchmark</title>
<script>document.documentElement.dataset.theme=localStorage.getItem("theme")||"light"</script>
<style>
:root{--bg:#ffffff;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
--muted:#656d76;--accent:#0969da;--green:#1a7f37;--red:#cf222e;
--radius:8px;--btn-text:#ffffff}
[data-theme="dark"]{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--btn-text:#0d1117}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh;
display:flex;align-items:center;justify-content:center}
.container{max-width:480px;width:100%;padding:24px}
h1{font-size:1.4rem;font-weight:600;margin-bottom:4px}
.subtitle{color:var(--muted);margin-bottom:32px;font-size:0.95rem}
.card{background:var(--surface);border:1px solid var(--border);
border-radius:var(--radius);padding:24px;margin-bottom:16px}
.card h2{font-size:1rem;margin-bottom:12px;color:var(--accent)}
input[type=text]{width:100%;padding:10px 14px;background:var(--bg);
border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:1rem;outline:none}
input[type=text]:focus{border-color:var(--accent);
box-shadow:0 0 0 3px rgba(88,166,255,0.15)}
input[type=text]::placeholder{color:var(--muted)}
button{display:block;width:100%;padding:10px 20px;margin-top:12px;
background:var(--accent);color:var(--btn-text);border:none;border-radius:6px;
font-size:1rem;font-weight:600;cursor:pointer;transition:opacity 0.15s}
button:hover{opacity:0.9}
button:disabled{opacity:0.4;cursor:not-allowed}
.error{color:var(--red);font-size:0.85rem;margin-top:8px;display:none}
.divider{text-align:center;color:var(--muted);margin:20px 0;font-size:0.85rem}
.info{font-size:0.85rem;color:var(--muted);margin-top:24px;text-align:center}
.theme-toggle{position:fixed;top:12px;right:12px;background:var(--surface);
border:1px solid var(--border);border-radius:50%;width:32px;height:32px;
cursor:pointer;display:flex;align-items:center;justify-content:center;
font-size:1rem;z-index:100;padding:0;color:var(--text);transition:all 0.2s}
.theme-toggle:hover{border-color:var(--accent)}
.theme-toggle::before{content:"\263E"}
[data-theme="dark"] .theme-toggle::before{content:"\2600"}
</style>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
<div class="container">
<h1>Human Evaluation</h1>
<p class="subtitle">Text Summarization Benchmark</p>
<div class="card">
<h2>Continue evaluation</h2>
<input type="text" id="token-input" placeholder="Paste your reviewer token"
  onkeydown="if(event.key==='Enter')continueEval()">
<div class="error" id="token-error"></div>
<button onclick="continueEval()">Continue</button>
</div>
<div class="divider">— or —</div>
<div class="card">
<h2>Start new evaluation</h2>
<input type="text" id="name-input" placeholder="Your name"
  onkeydown="if(event.key==='Enter')register()">
<div class="error" id="register-error"></div>
<button onclick="register()">Start</button>
</div>
<p class="info">__ASSESSMENT_INFO__</p>
</div>
<script>
function toggleTheme(){
  var t=document.documentElement.dataset.theme==='dark'?'light':'dark';
  document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);
}
async function continueEval(){
  const token=document.getElementById('token-input').value.trim();
  const err=document.getElementById('token-error');
  err.style.display='none';
  if(!token){err.textContent='Please enter your token.';err.style.display='block';return}
  const r=await fetch('/api/progress?token='+encodeURIComponent(token));
  if(!r.ok){err.textContent='Token not found.';err.style.display='block';return}
  window.location.href='/evaluate?token='+encodeURIComponent(token);
}
async function register(){
  const name=document.getElementById('name-input').value.trim();
  const err=document.getElementById('register-error');
  err.style.display='none';
  if(!name){err.textContent='Please enter your name.';err.style.display='block';return}
  const r=await fetch('/api/register',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  if(!r.ok){err.textContent='Registration failed.';err.style.display='block';return}
  const d=await r.json();
  window.location.href='/evaluate?token='+encodeURIComponent(d.token);
}
</script>
</body>
</html>"""

EVALUATE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation – Text Summarization Benchmark</title>
<script>document.documentElement.dataset.theme=localStorage.getItem("theme")||"light"</script>
<style>
:root{--bg:#ffffff;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
--muted:#656d76;--accent:#0969da;--green:#1a7f37;--red:#cf222e;
--yellow:#9a6700;--radius:8px;--btn-text:#ffffff}
[data-theme="dark"]{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;
--yellow:#d29922;--btn-text:#0d1117}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.6}
.container{max-width:800px;margin:0 auto;padding:16px 16px 32px}
header{margin-bottom:14px}
h1{font-size:1.1rem;font-weight:600;margin-bottom:8px}
.progress-wrap{background:var(--surface);border:1px solid var(--border);
border-radius:20px;height:20px;overflow:hidden;margin-bottom:4px;position:relative}
.progress-bar{height:100%;background:var(--accent);border-radius:20px;
transition:width 0.4s ease}
.progress-text{position:absolute;inset:0;display:flex;align-items:center;
justify-content:center;font-size:0.7rem;font-weight:600;color:var(--text);
mix-blend-mode:difference;pointer-events:none}
.card{background:var(--surface);border:1px solid var(--border);
border-radius:var(--radius);padding:14px;margin-bottom:10px}
.card-header{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;
color:var(--accent);margin-bottom:6px;font-weight:600}
.highlights{list-style:none;padding:0;margin:4px 0 0}
.highlights li{position:relative;padding:3px 10px 3px 20px;margin-bottom:3px;
font-size:0.85rem;line-height:1.5;color:var(--text);
background:var(--bg);border:1px solid var(--border);border-radius:4px}
.highlights li::before{content:"\2022";position:absolute;left:8px;
color:var(--accent);font-weight:700}
.gen-text{background:var(--bg);border:1px solid var(--border);
border-radius:6px;padding:10px;font-size:0.85rem;line-height:1.5;
color:var(--text);white-space:pre-wrap}
.criterion{margin-bottom:10px}
.criterion-header{display:flex;align-items:baseline;gap:6px;margin-bottom:4px;
flex-wrap:wrap}
.criterion-label{font-weight:600;font-size:0.88rem}
.criterion-desc{font-size:0.78rem;color:var(--muted)}
.likert{display:flex;gap:4px}
.likert label{flex:1;text-align:center;padding:6px 2px;
border:2px solid var(--border);border-radius:6px;cursor:pointer;
font-size:0.8rem;transition:all 0.15s;user-select:none;line-height:1.2}
.likert label:hover{border-color:var(--accent);
background:rgba(88,166,255,0.08)}
.likert input{display:none}
.likert label.selected{border-color:var(--accent);
background:var(--accent);color:var(--btn-text);font-weight:600}
.likert .anchor{display:block;font-size:0.62rem;margin-top:1px;opacity:0.8}
.submit-row{display:flex;justify-content:flex-end;margin-top:10px}
button{padding:10px 28px;background:var(--accent);color:var(--btn-text);border:none;
border-radius:8px;font-size:0.95rem;font-weight:600;cursor:pointer;
transition:opacity 0.15s}
button:hover{opacity:0.9}
button:disabled{opacity:0.35;cursor:not-allowed}
.done-card{text-align:center;padding:48px 24px}
.done-card h2{font-size:1.3rem;color:var(--green);margin-bottom:12px}
.done-card p{color:var(--muted)}
.token-bar{display:flex;align-items:center;justify-content:space-between;
margin-bottom:20px;padding:10px 14px;background:var(--surface);
border:1px solid var(--border);border-radius:var(--radius);font-size:0.82rem}
.token-bar code{color:var(--accent);user-select:all}
.token-bar .copy-btn{background:none;border:1px solid var(--border);
color:var(--muted);padding:4px 10px;font-size:0.78rem;border-radius:4px;
cursor:pointer;margin:0}
.token-bar .copy-btn:hover{border-color:var(--accent);color:var(--accent)}
.loading{text-align:center;padding:48px;color:var(--muted)}
.theme-toggle{position:fixed;top:12px;right:12px;background:var(--surface);
border:1px solid var(--border);border-radius:50%;width:32px;height:32px;
cursor:pointer;display:flex;align-items:center;justify-content:center;
font-size:1rem;z-index:100;padding:0;color:var(--text);transition:all 0.2s}
.theme-toggle:hover{border-color:var(--accent)}
.theme-toggle::before{content:"\263E"}
[data-theme="dark"] .theme-toggle::before{content:"\2600"}
.info-btn{display:inline-flex;align-items:center;justify-content:center;
width:20px;height:20px;border-radius:50%;background:transparent;
border:1.5px solid var(--muted);color:var(--muted);font-size:0.72rem;
font-weight:700;cursor:pointer;margin-left:8px;transition:all 0.15s;
vertical-align:middle;line-height:1;font-style:italic;font-family:Georgia,serif;
padding:0}
.info-btn:hover{border-color:var(--accent);color:var(--accent)}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);
z-index:1000;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);
border-radius:var(--radius);max-width:600px;width:90%;max-height:80vh;
overflow-y:auto;padding:24px}
.modal h3{font-size:1rem;margin-bottom:16px;color:var(--accent)}
.modal h4{font-size:0.9rem;font-weight:600;margin:16px 0 4px;color:var(--text)}
.modal p{font-size:0.85rem;color:var(--muted);line-height:1.6;margin-bottom:8px}
.modal .close-btn{display:block;margin:20px auto 0;padding:8px 24px;
background:var(--border);color:var(--text);border:none;border-radius:6px;
font-size:0.85rem;cursor:pointer}
.modal .close-btn:hover{background:var(--accent);color:var(--btn-text)}
.modal .ref{font-size:0.78rem;color:var(--muted);margin-top:16px;
border-top:1px solid var(--border);padding-top:12px;font-style:italic}
.nav-grid{display:flex;flex-wrap:wrap;gap:2px;margin:8px 0 2px;padding:6px;
background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)}
.nav-item{width:22px;height:22px;display:flex;align-items:center;justify-content:center;
border:1.5px solid var(--border);border-radius:4px;font-size:0.6rem;font-weight:600;
cursor:pointer;transition:all 0.15s;user-select:none;background:var(--bg)}
.nav-item:hover:not(.locked){border-color:var(--accent);background:rgba(88,166,255,0.08)}
.nav-item.done{background:var(--green);color:var(--btn-text);border-color:var(--green)}
.nav-item.active{border-color:var(--accent);box-shadow:0 0 0 1.5px rgba(88,166,255,0.3)}
.nav-item.done.active{box-shadow:0 0 0 1.5px rgba(59,185,80,0.4)}
.nav-item.locked{opacity:0.35;cursor:default}
.nav-legend{display:flex;gap:10px;font-size:0.68rem;color:var(--muted);margin-bottom:6px;
padding-left:6px}
.nav-legend-item{display:flex;align-items:center;gap:3px}
.nav-swatch{width:8px;height:8px;border-radius:2px;border:1.5px solid var(--border)}
.nav-swatch.sw-done{background:var(--green);border-color:var(--green)}
.nav-swatch.sw-current{border-color:var(--accent);box-shadow:0 0 0 1px rgba(88,166,255,0.3)}
.nav-swatch.sw-pending{background:var(--bg)}
.status-badge{font-size:0.7rem;text-transform:none;letter-spacing:0;
padding:2px 8px;border-radius:4px;margin-left:8px;font-weight:600}
.status-badge.completed{color:var(--green);background:rgba(26,127,55,0.1)}
textarea{width:100%;min-height:60px;padding:8px 10px;background:var(--bg);
border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:0.85rem;font-family:inherit;resize:vertical;outline:none}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.step-indicator{font-size:0.72rem;color:var(--muted);font-weight:600;
margin-left:8px;padding:2px 8px;background:var(--bg);border:1px solid var(--border);
border-radius:4px}
.abstract-text{background:var(--bg);border:1px solid var(--border);
border-radius:6px;padding:10px;font-size:0.85rem;line-height:1.5;
color:var(--text);white-space:pre-wrap}
.btn-row{display:flex;justify-content:space-between;margin-top:10px}
.btn-back{padding:10px 28px;background:var(--surface);color:var(--text);
border:1px solid var(--border);border-radius:8px;font-size:0.95rem;
font-weight:600;cursor:pointer;transition:all 0.15s}
.btn-back:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>
<div class="modal-overlay" id="info-modal" onclick="if(event.target===this)closeInfo()">
<div class="modal" id="info-modal-body"></div>
</div>
<button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
<div class="container">
<header>
<h1>Text Summarization Evaluation</h1>
<div class="token-bar">
  <span>Your token: <code id="token-display"></code></span>
  <button class="copy-btn" onclick="copyToken(this)">Copy</button>
</div>
<div class="progress-wrap"><div class="progress-bar" id="pbar"></div><div class="progress-text" id="ptext"></div></div>
<div class="nav-grid" id="nav-grid"></div>
<div class="nav-legend">
<span class="nav-legend-item"><span class="nav-swatch sw-done"></span>Completed</span>
<span class="nav-legend-item"><span class="nav-swatch sw-current"></span>Current</span>
<span class="nav-legend-item"><span class="nav-swatch sw-pending"></span>Pending</span>
</div>
</header>
<main id="main"><div class="loading">Loading…</div></main>
</div>
<script>
function toggleTheme(){
  var t=document.documentElement.dataset.theme==='dark'?'light':'dark';
  document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);
}
const TOKEN=new URLSearchParams(window.location.search).get('token');
function copyToken(btn){
  function ok(){btn.textContent='Copied!';setTimeout(function(){btn.textContent='Copy'},1500)}
  function no(){btn.textContent='Failed';setTimeout(function(){btn.textContent='Copy'},1500)}
  if(navigator.clipboard&&window.isSecureContext){
    navigator.clipboard.writeText(TOKEN).then(ok,no);
  }else{
    try{var ta=document.createElement('textarea');ta.value=TOKEN;
    ta.style.position='fixed';ta.style.opacity='0';
    document.body.appendChild(ta);ta.select();document.execCommand('copy');
    document.body.removeChild(ta);ok();}catch(e){no();}
  }
}
const CRITERIA=__CRITERIA_PLACEHOLDER__;
const STEP1_CRITERIA=CRITERIA.filter(c=>c.key!=='consistency');
const STEP2_CRITERIA=CRITERIA.filter(c=>c.key==='consistency');
document.getElementById('token-display').textContent=TOKEN;
let currentData=null;
let overviewData=null;
let currentStep=1;
let pendingRatings={};
let pendingComment='';

async function loadOverview(){
  const r=await fetch('/api/overview?token='+encodeURIComponent(TOKEN));
  if(r.ok) overviewData=await r.json();
  renderNavGrid();
}

function renderNavGrid(){
  const grid=document.getElementById('nav-grid');
  if(!overviewData||!grid)return;
  const activeIdx=currentData&&!currentData.done?currentData.assessment_index:-1;
  let html='';
  for(const item of overviewData.items){
    const done=item.is_completed;
    const active=item.index===activeIdx;
    const reachable=item.index<=overviewData.completed;
    let cls='nav-item';
    if(done)cls+=' done';
    if(active)cls+=' active';
    if(!reachable&&!done)cls+=' locked';
    const tip=escH(item.paper_title||'').substring(0,60)+(item.model_label?' \u2014 '+escH(item.model_label):'');
    html+='<div class="'+cls+'"'+(reachable?' onclick="goTo('+item.index+')"':'')+
      ' title="'+tip+'">'+(item.index+1)+'</div>';
  }
  grid.innerHTML=html;
}

async function goTo(index){
  currentStep=1;pendingRatings={};pendingComment='';
  const r=await fetch('/api/assessment?token='+encodeURIComponent(TOKEN)+'&index='+index);
  if(!r.ok)return;
  const d=await r.json();
  currentData=d;
  updateProgress(d.completed,d.total);
  if(d.done){renderDone(d);}
  else{initAssessment(d);renderStep1(d);}
  loadOverview();
}

async function loadAssessment(){
  currentStep=1;pendingRatings={};pendingComment='';
  const r=await fetch('/api/assessment?token='+encodeURIComponent(TOKEN));
  if(!r.ok){document.getElementById('main').innerHTML=
    '<div class="card"><p>Error loading assessment. Is your token valid?</p></div>';return}
  const d=await r.json();
  currentData=d;
  updateProgress(d.completed,d.total);
  if(d.done){renderDone(d);}
  else{initAssessment(d);renderStep1(d);}
  loadOverview();
}

function initAssessment(d){
  if(d.previous_ratings){
    pendingRatings=Object.assign({},d.previous_ratings);
    pendingComment=d.previous_comment||'';
  }
}

function updateProgress(completed,total){
  const pct=total>0?Math.round(completed/total*100):0;
  document.getElementById('pbar').style.width=pct+'%';
  document.getElementById('ptext').textContent=completed+' / '+total+' completed ('+pct+'%)';
}

function renderDone(d){
  document.getElementById('main').innerHTML=
    '<div class="card done-card"><h2>All done!</h2>'+
    '<p>You have completed all '+d.total+' assessments.</p>'+
    '<p style="margin-top:8px;color:var(--muted);font-size:0.85rem">'+
    'Click any item in the grid above to review or change your answers.</p>'+
    '<p style="margin-top:12px;color:var(--text)">Thank you for your time, '+
    escH(d.reviewer_name)+'.</p></div>';
}

function buildCriteriaHTML(criteriaList){
  let html='';
  for(const c of criteriaList){
    html+='<div class="criterion"><div class="criterion-header">'+
      '<span class="criterion-label">'+escH(c.label)+'</span>'+
      '<span class="criterion-desc">'+escH(c.description)+'</span></div><div class="likert">';
    for(let v=1;v<=5;v++){
      const anchor=c.anchors[String(v)]||'';
      const sel=pendingRatings[c.key]===v?' selected':'';
      html+='<label class="'+sel.trim()+'" data-key="'+c.key+'" data-val="'+v+'" onclick="selectRating(this)">'+
        v+(anchor?'<span class="anchor">'+escH(anchor)+'</span>':'')+'</label>';
    }
    html+='</div></div>';
  }
  return html;
}

function renderStep1(d){
  currentStep=1;
  const bullets=d.reference_highlights.map(h=>'<li>'+escH(h)+'</li>').join('');
  const statusBadge=d.is_submitted?'<span class="status-badge completed">\u2714 completed</span>':'';
  const stepTag='<span class="step-indicator">Step 1 of 2</span>';
  const criteriaHTML=buildCriteriaHTML(STEP1_CRITERIA);
  document.getElementById('main').innerHTML=
    '<div class="card"><div class="card-header">Assessment '+(d.assessment_index+1)+' of '+d.total+
    statusBadge+stepTag+'</div>'+
    '<div class="card-header" style="margin-top:8px">Reference Highlights</div>'+
    '<ul class="highlights">'+bullets+'</ul></div>'+
    '<div class="card"><div class="card-header">Generated Summary</div>'+
    '<div class="gen-text">'+escH(d.summary)+'</div></div>'+
    '<div class="card"><div class="card-header">Rate Structure &amp; Content'+
    ' <button class="info-btn" onclick="openInfo()" title="About these criteria">i</button></div>'+
    criteriaHTML+
    '<div class="submit-row"><button id="next-btn" onclick="goToStep2()" disabled>Next \u2192</button></div></div>';
  checkStep1();
}

function renderStep2(d){
  currentStep=2;
  const isUpdate=d.is_submitted;
  const btnLabel=isUpdate?'Update':'Submit';
  const statusBadge=isUpdate?'<span class="status-badge completed">\u2714 completed</span>':'';
  const stepTag='<span class="step-indicator">Step 2 of 2</span>';
  const criteriaHTML=buildCriteriaHTML(STEP2_CRITERIA);
  document.getElementById('main').innerHTML=
    '<div class="card"><div class="card-header">Assessment '+(d.assessment_index+1)+' of '+d.total+
    statusBadge+stepTag+'</div>'+
    '<div style="margin-bottom:10px"><div class="card-header">Publication Title</div>'+
    '<div class="gen-text">'+escH(d.paper_title)+'</div></div>'+
    '<div style="margin-bottom:10px"><div class="card-header">Abstract</div>'+
    '<div class="abstract-text">'+escH(d.paper_abstract)+'</div></div>'+
    '<div><div class="card-header">Generated Summary</div>'+
    '<div class="gen-text">'+escH(d.summary)+'</div></div></div>'+
    '<div class="card"><div class="card-header">Rate Factual Consistency'+
    ' <button class="info-btn" onclick="openInfo()" title="About these criteria">i</button></div>'+
    criteriaHTML+
    '<div style="margin-top:8px"><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:4px">Comment (optional)</label>'+
    '<textarea id="comment" placeholder="Any observations\u2026"></textarea></div>'+
    '<div class="btn-row"><button class="btn-back" onclick="goToStep1()">\u2190 Back</button>'+
    '<button id="submit-btn" onclick="submitAssessment()" disabled>'+btnLabel+'</button></div></div>';
  const ta=document.getElementById('comment');
  if(ta&&pendingComment)ta.value=pendingComment;
  checkStep2();
}

function goToStep2(){
  // Save Step 1 ratings
  for(const c of STEP1_CRITERIA){
    const sel=document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
    if(sel)pendingRatings[c.key]=parseInt(sel.dataset.val);
  }
  renderStep2(currentData);
}

function goToStep1(){
  // Save Step 2 state
  const sel=document.querySelector('.likert label.selected[data-key="consistency"]');
  if(sel)pendingRatings.consistency=parseInt(sel.dataset.val);
  const ta=document.getElementById('comment');
  if(ta)pendingComment=ta.value;
  renderStep1(currentData);
}

function selectRating(el){
  const key=el.dataset.key;
  el.parentElement.querySelectorAll('label').forEach(l=>l.classList.remove('selected'));
  el.classList.add('selected');
  if(currentStep===1)checkStep1();else checkStep2();
}

function checkStep1(){
  const btn=document.getElementById('next-btn');
  if(!btn)return;
  btn.disabled=!STEP1_CRITERIA.every(c=>
    document.querySelector('.likert label.selected[data-key="'+c.key+'"]'));
}

function checkStep2(){
  const btn=document.getElementById('submit-btn');
  if(!btn)return;
  btn.disabled=!STEP2_CRITERIA.every(c=>
    document.querySelector('.likert label.selected[data-key="'+c.key+'"]'));
}

async function submitAssessment(){
  const btn=document.getElementById('submit-btn');
  const wasUpdate=currentData.is_submitted;
  btn.disabled=true;btn.textContent='Saving\u2026';
  // Collect Step 2 consistency
  for(const c of STEP2_CRITERIA){
    const sel=document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
    if(sel)pendingRatings[c.key]=parseInt(sel.dataset.val);
  }
  const comment=(document.getElementById('comment')||{}).value||'';
  const body={token:TOKEN,assessment_index:currentData.assessment_index,
    ratings:Object.assign({},pendingRatings),comment};
  const r=await fetch('/api/submit',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){btn.textContent='Error \u2013 retry';btn.disabled=false;return}
  if(wasUpdate){
    btn.textContent='Updated \u2713';
    setTimeout(function(){btn.textContent='Update';btn.disabled=false;},1200);
    loadOverview();
  }else{
    loadAssessment();
  }
}

function openInfo(){
  const ref='<p class="ref">Fabbri, A.&nbsp;R. et&nbsp;al. (2021). SummEval: Re-evaluating Summarization Evaluation. <em>Transactions of the Association for Computational Linguistics</em>, 9, 391&ndash;409.</p>';
  let html;
  if(currentStep===1){
    html='<h3>Step 1 Criteria &mdash; Structure &amp; Content</h3>'+
      '<p>Compare the generated summary against the <strong>reference highlights</strong> shown above. Rate each dimension on a 1&ndash;5 Likert scale.</p>'+
      '<h4>Coherence</h4><p>The summary should be well-structured and well-organized. It should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.</p>'+
      '<h4>Fluency</h4><p>The quality of individual sentences. They should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read.</p>'+
      '<h4>Relevance</h4><p>Does the summary capture the important information present in the reference highlights? Penalize summaries which contain redundancies, miss key points from the highlights, or include excess information.</p>'+
      ref;
  }else{
    html='<h3>Step 2 Criteria &mdash; Factual Consistency</h3>'+
      '<p>Compare the generated summary against the <strong>publication title and abstract</strong> shown above. Rate whether the summary is factually consistent with the source.</p>'+
      '<h4>Consistency</h4><p>The factual alignment between the summary and the title and abstract. A factually consistent summary contains only statements that are entailed by the source. Penalize summaries that contain hallucinated facts not supported by the title or abstract.</p>'+
      ref;
  }
  html+='<button class="close-btn" onclick="closeInfo()">Close</button>';
  document.getElementById('info-modal-body').innerHTML=html;
  document.getElementById('info-modal').classList.add('open');
}
function closeInfo(){document.getElementById('info-modal').classList.remove('open')}

function escH(s){
  const d=document.createElement('div');d.textContent=s;return d.innerHTML;
}

loadAssessment();
</script>
</body>
</html>"""


# ── HTTP handler ─────────────────────────────────────────────────────────


def make_handler(
    eval_data: dict,
    data_dir: Path,
    rating_mode: str,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/":
                self._serve_landing()
            elif path == "/evaluate":
                token = (params.get("token") or [None])[0]
                if not token or load_reviewer(data_dir, token) is None:
                    self._redirect("/")
                else:
                    self._serve_evaluate()
            elif path == "/api/assessment":
                self._handle_get_assessment(params)
            elif path == "/api/progress":
                self._handle_get_progress(params)
            elif path == "/api/overview":
                self._handle_overview(params)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            body = self._read_body()
            if body is None:
                self._send_json({"error": "invalid or oversized JSON body"}, 400)
                return

            if path == "/api/register":
                self._handle_register(body)
            elif path == "/api/submit":
                self._handle_submit(body)
            else:
                self._send_json({"error": "not found"}, 404)

        # ── Page handlers ────────────────────────────────────────

        def _serve_landing(self):
            n = len(eval_data["papers"])
            m = len(eval_data["models"])
            info = html_module.escape(
                f"{n} papers \u00d7 {m} models"
                f" = {n * m} assessments per reviewer"
            )
            page = LANDING_HTML.replace("__ASSESSMENT_INFO__", info)
            self._send_html(page)

        def _serve_evaluate(self):
            criteria_json = (
                json.dumps(DETAILED_CRITERIA)
                .replace("</", "<\\/")
                .replace("<!--", "<\\!--")
            )
            page = EVALUATE_HTML.replace(
                "__CRITERIA_PLACEHOLDER__", criteria_json
            )
            self._send_html(page)

        # ── API handlers ─────────────────────────────────────────

        def _handle_get_assessment(self, params):
            token = (params.get("token") or [None])[0]
            reviewer = load_reviewer(data_dir, token) if token else None
            if not reviewer:
                self._send_json({"error": "invalid token"}, 404)
                return
            completed = len(reviewer["assessments"])
            total = len(reviewer["assignments"])

            # Optional index param for navigating to a specific assessment
            idx_str = (params.get("index") or [None])[0]
            if idx_str is not None:
                try:
                    idx = int(idx_str)
                except (ValueError, TypeError):
                    self._send_json(
                        {"error": "index must be an integer"}, 400
                    )
                    return
                if idx < 0 or idx >= total:
                    self._send_json(
                        {"error": "index out of range"}, 400
                    )
                    return
                if idx > completed:
                    self._send_json(
                        {"error": "cannot skip ahead past current progress"},
                        400,
                    )
                    return
            else:
                idx = completed

            if idx >= total:
                self._send_json(
                    {
                        "done": True,
                        "completed": completed,
                        "total": total,
                        "reviewer_name": reviewer["name"],
                    }
                )
                return

            assignment = reviewer["assignments"][idx]
            paper = eval_data["papers"][assignment["paper_index"]]
            previous = reviewer["assessments"][idx] if idx < completed else None

            summary = paper["summaries"][assignment["model"]]
            resp = {
                "done": False,
                "completed": completed,
                "total": total,
                "assessment_index": idx,
                "is_submitted": idx < completed,
                "paper_title": paper["title"],
                "paper_abstract": paper["abstract"],
                "reference_highlights": paper["reference_highlights"],
                "summary": summary,
                "reviewer_name": reviewer["name"],
            }
            if previous:
                resp["previous_ratings"] = previous.get("ratings")
                resp["previous_comment"] = previous.get("comment", "")

            self._send_json(resp)

        def _handle_get_progress(self, params):
            token = (params.get("token") or [None])[0]
            reviewer = load_reviewer(data_dir, token) if token else None
            if not reviewer:
                self._send_json({"error": "invalid token"}, 404)
                return
            self._send_json(
                {
                    "token": reviewer["token"],
                    "name": reviewer["name"],
                    "completed": len(reviewer["assessments"]),
                    "total": len(reviewer["assignments"]),
                }
            )

        def _handle_overview(self, params):
            token = (params.get("token") or [None])[0]
            reviewer = load_reviewer(data_dir, token) if token else None
            if not reviewer:
                self._send_json({"error": "invalid token"}, 404)
                return

            completed = len(reviewer["assessments"])
            items = []
            for i, assignment in enumerate(reviewer["assignments"]):
                paper = eval_data["papers"][assignment["paper_index"]]
                items.append({
                    "index": i,
                    "is_completed": i < completed,
                    "paper_title": paper["title"],
                    "model_label": assignment.get("model", ""),
                })

            self._send_json(
                {
                    "completed": completed,
                    "total": len(reviewer["assignments"]),
                    "items": items,
                    "reviewer_name": reviewer["name"],
                }
            )

        def _handle_register(self, body: dict):
            name = (body.get("name") or "").strip()
            if not name:
                self._send_json({"error": "name is required"}, 400)
                return
            if len(name) > MAX_NAME_LENGTH:
                self._send_json(
                    {"error": f"name must be {MAX_NAME_LENGTH} characters or fewer"}, 400
                )
                return
            reviewer = create_reviewer(
                data_dir, name, eval_data, rating_mode
            )
            self._send_json({"token": reviewer["token"]})

        def _handle_submit(self, body: dict):
            token = body.get("token")
            idx = body.get("assessment_index")
            comment = str(body.get("comment", ""))[:MAX_COMMENT_LENGTH]

            if not isinstance(idx, int):
                self._send_json(
                    {"error": "assessment_index must be an integer"}, 400
                )
                return

            # ── Validation ──────────────────────────────────────────
            ratings = body.get("ratings", {})
            if not isinstance(ratings, dict):
                self._send_json(
                    {"error": "ratings must be an object"}, 400
                )
                return
            required_keys = {c["key"] for c in DETAILED_CRITERIA}
            provided_keys = set(ratings.keys())
            if not required_keys.issubset(provided_keys):
                missing = required_keys - provided_keys
                self._send_json(
                    {"error": f"missing ratings: {missing}"}, 400
                )
                return
            for key in required_keys:
                val = ratings[key]
                if not isinstance(val, int) or val < 1 or val > 5:
                    self._send_json(
                        {"error": f"rating '{key}' must be 1\u20135"},
                        400,
                    )
                    return

            # ── Atomic read-modify-write under lock ──────────────
            with _write_lock:
                reviewer = load_reviewer(data_dir, token) if token else None
                if not reviewer:
                    self._send_json({"error": "invalid token"}, 404)
                    return

                completed = len(reviewer["assessments"])
                if idx > completed:
                    self._send_json(
                        {
                            "error": f"index {idx} is beyond current "
                            f"progress ({completed})"
                        },
                        400,
                    )
                    return

                assignment = reviewer["assignments"][idx]
                paper = eval_data["papers"][assignment["paper_index"]]

                now = datetime.now(timezone.utc).isoformat()
                entry = {
                    "paper_id": paper["id"],
                    "model": assignment["model"],
                    "ratings": {
                        k: int(ratings[k]) for k in required_keys
                    },
                    "comment": comment,
                    "submitted_at": now,
                }

                if idx < completed:
                    # Preserve original timestamp, record update time
                    original = reviewer["assessments"][idx]
                    entry["originally_submitted_at"] = original.get(
                        "originally_submitted_at",
                        original.get("submitted_at"),
                    )
                    entry["submitted_at"] = now
                    reviewer["assessments"][idx] = entry
                else:
                    reviewer["assessments"].append(entry)
                _write_reviewer(data_dir, reviewer)

                new_completed = len(reviewer["assessments"])

            log.info(
                "Reviewer %s (%s): %d/%d",
                reviewer["token"],
                reviewer["name"],
                new_completed,
                len(reviewer["assignments"]),
            )
            self._send_json({"ok": True, "completed": new_completed})

        # ── Utilities ────────────────────────────────────────────

        def _read_body(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                return None
            if length < 0 or length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        def _send_html(self, html: str, status: int = 200):
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: dict, status: int = 200):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str):
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, fmt, *args):
            log.debug("HTTP %s", fmt % args)

    return Handler


# ── CLI & main ───────────────────────────────────────────────────────────


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "models",
        nargs=4,
        metavar="MODEL",
        help="exactly 4 model names (as they appear in the results file)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"server port (default: {DEFAULT_PORT})",
    )
    p.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS,
        help="path to detailed_scores_per_paper.json",
    )
    p.add_argument(
        "--goldstandard",
        type=Path,
        default=DEFAULT_GOLDSTANDARD,
        help="path to gold-standard dataset JSON",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory for reviewer JSON files",
    )
    p.add_argument(
        "--num-papers",
        type=int,
        default=DEFAULT_NUM_PAPERS,
        help=f"papers to select (default: {DEFAULT_NUM_PAPERS})",
    )
    return p.parse_args()


def main():
    args = cli()

    rating_mode = "detailed"

    eval_data = load_evaluation_data(
        args.results_file, args.goldstandard, args.models, args.num_papers
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)

    handler = make_handler(eval_data, args.data_dir, rating_mode)
    server = ThreadingHTTPServer(("", args.port), handler)

    log.info("=" * 56)
    log.info("Human Evaluation Server")
    log.info("-" * 56)
    log.info("Port:         %d", args.port)
    log.info("Models:       %s", ", ".join(args.models))
    log.info("Rating mode:  %s", rating_mode)
    log.info("Papers:       %d", len(eval_data["papers"]))
    log.info("Assessments:  %d per reviewer", eval_data["total_assessments"])
    log.info("Data dir:     %s", args.data_dir)
    log.info("-" * 56)
    log.info("Server running at http://localhost:%d", args.port)
    log.info("=" * 56)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
