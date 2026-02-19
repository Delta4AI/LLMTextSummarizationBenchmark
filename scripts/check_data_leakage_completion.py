#!/usr/bin/env python3
"""
Data-leakage probe: Abstract completion test.

For each at-risk model, prompt with the first sentence of sample abstracts
and ask the model to continue the abstract.  Compare the model's completion
against the actual remaining text using ROUGE metrics.  High similarity
suggests the abstract may have been memorized during training.

Sample selection (per journal):
  - 1 article with the EARLIEST publication date   → "oldest"
  - 1 article with the LATEST  publication date    → "newest"
  → 40 articles total across 20 journals.

For each model the articles are classified as *pre_cutoff* (published ≤
model's training cutoff → could be in training data) or *post_cutoff*
(published after → safe).  A large gap in ROUGE scores between the two
groups is evidence of memorization.

Outputs:
  Resources/data_leakage_completion_probe.json
  Resources/data_leakage_completion_probe.html

Expects API keys in Resources/.env (see Resources/example.env).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nltk
from dotenv import load_dotenv
from rouge_score import rouge_scorer as rs

# ── paths & env ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
for _env in [ROOT / "Resources" / ".env", ROOT / ".env"]:
    if _env.exists():
        load_dotenv(_env, override=False)
        break

LEAKAGE_JSON = ROOT / "Resources" / "data_leakage_cutoff_check.json"
DATASET_JSON = ROOT / "Resources" / "text_summarization_goldstandard_data.json"
CACHE_PATH   = ROOT / "Resources" / "data_leakage_completion_cache.json"
OUTPUT_JSON  = ROOT / "Resources" / "data_leakage_completion_probe.json"
OUTPUT_HTML  = ROOT / "Resources" / "data_leakage_completion_probe.html"

# ── configuration ────────────────────────────────────────────────────────────

# Only test models whose publication-date overlap exceeds this threshold
MIN_OVERLAP_PCT = 5.0

# Platforms for which we have API access
TESTABLE_PLATFORMS = {"openai", "anthropic", "ollama"}
CLOSED_PLATFORMS   = {"openai", "anthropic", "mistral"}

SYSTEM_PROMPT = (
    "You are given the first sentence of a published scientific abstract. "
    "Continue and complete the abstract from where it left off. "
    "Output ONLY the continuation text — no preamble, no labels, no commentary."
)
USER_PROMPT_TEMPLATE = (
    'First sentence of a scientific abstract:\n\n'
    '"{first_sentence}"\n\n'
    'Continuation:'
)

TEMPERATURE  = 0.0          # deterministic for memorization detection
MAX_TOKENS   = 1024
REQUEST_DELAY = 0.5         # seconds between API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_partial_date(d: str | None) -> datetime | None:
    if d is None:
        return None
    d = str(d).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def get_journal_id(url: str) -> str:
    if "cell.com" in url:
        m = re.search(r"cell\.com/([^/]+)/", url)
        return m.group(1) if m else "cell-unknown"
    if "sciencedirect.com" in url:
        m = re.search(r"pii/(S\d{9})", url)
        return f"sd-{m.group(1)}" if m else "sd-unknown"
    return "unknown"


def extract_first_sentence(text: str) -> tuple[str, str]:
    """Return (first_sentence, remaining_text) using NLTK tokeniser."""
    try:
        sents = nltk.sent_tokenize(text)
        if len(sents) >= 2:
            return sents[0], " ".join(sents[1:])
    except Exception:
        pass
    # fallback
    parts = text.split(". ", 1)
    if len(parts) == 2:
        return parts[0] + ".", parts[1]
    return text, ""


def clean_completion(raw: str, first_sentence: str) -> str:
    """Strip echoed prompt, preamble, thinking tags, and quotes."""
    text = raw.strip()
    # DeepSeek-R1 thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # common preambles
    for prefix in [
        "Here is the continuation of the abstract:",
        "Here's the continuation of the abstract:",
        "Here is the continuation:",
        "Here's the continuation:",
        "Continuation:",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # echoed first sentence
    if text.startswith(first_sentence):
        text = text[len(first_sentence):].strip()
    # surrounding quotes
    if len(text) > 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def _cache_key(model: str, article_id: str) -> str:
    raw = f"{model}::{article_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


# ── cache ────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ── API clients (lazy, singleton) ────────────────────────────────────────────

_clients: dict = {}


def _get_openai():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _clients["openai"]


def _get_anthropic():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    return _clients["anthropic"]


def _get_ollama():
    if "ollama" not in _clients:
        import ollama
        _clients["ollama"] = ollama.Client(
            host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    return _clients["ollama"]


def query_model(platform: str, model: str, prompt: str) -> str:
    """Send a completion probe to the given model and return the raw text."""
    if platform == "openai":
        client = _get_openai()
        if model.startswith("gpt-5"):
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                text={"verbosity": "low"},
                reasoning={"effort": "minimal"},
            )
        else:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                temperature=TEMPERATURE,
            )
        return resp.output_text

    if platform == "anthropic":
        client = _get_anthropic()
        resp = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.content[0].text

    if platform == "ollama":
        client = _get_ollama()
        resp = client.generate(
            model=model,
            prompt=f"{SYSTEM_PROMPT}\n\n{prompt}",
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        )
        return resp["response"]

    raise ValueError(f"Unsupported platform: {platform}")


# ── scoring ──────────────────────────────────────────────────────────────────

_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def longest_common_substring_ratio(ref: str, hyp: str) -> float:
    """LCS-char ratio relative to the reference length."""
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best / m


def score_completion(reference: str, hypothesis: str) -> dict:
    scores = _scorer.score(reference, hypothesis)
    lcs = longest_common_substring_ratio(reference.lower(), hypothesis.lower())
    return {
        "rouge1_p": round(scores["rouge1"].precision, 4),
        "rouge1_r": round(scores["rouge1"].recall, 4),
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_p": round(scores["rouge2"].precision, 4),
        "rouge2_r": round(scores["rouge2"].recall, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_p": round(scores["rougeL"].precision, 4),
        "rougeL_r": round(scores["rougeL"].recall, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        "lcs_ratio": round(lcs, 4),
    }


# ── article selection ────────────────────────────────────────────────────────

def select_samples(articles: list[dict]) -> list[dict]:
    """Pick the oldest + newest article per journal."""
    journals: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        journals[get_journal_id(a["url"])].append(a)

    samples: list[dict] = []
    seen_ids: set[str] = set()
    for j_id in sorted(journals):
        by_date = sorted(journals[j_id], key=lambda a: a.get("publication_date", ""))
        for art, group in [(by_date[0], "oldest"), (by_date[-1], "newest")]:
            if art["id"] not in seen_ids:
                first, remaining = extract_first_sentence(art["abstract"])
                samples.append({
                    "article": art,
                    "journal_id": j_id,
                    "sample_group": group,
                    "first_sentence": first,
                    "remaining_text": remaining,
                    "pub_date": art.get("publication_date", "unknown"),
                })
                seen_ids.add(art["id"])
    return samples


# ── aggregation ──────────────────────────────────────────────────────────────

def _avg(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_model(probes: list[dict]) -> dict:
    """Compute per-model summary statistics."""
    pre  = [p for p in probes if p["temporal_class"] == "pre_cutoff"]
    post = [p for p in probes if p["temporal_class"] == "post_cutoff"]

    def _stats(subset, prefix):
        return {
            f"{prefix}_n": len(subset),
            f"{prefix}_rouge1_f": _avg([p["rouge1_f"] for p in subset]),
            f"{prefix}_rouge2_f": _avg([p["rouge2_f"] for p in subset]),
            f"{prefix}_rougeL_f": _avg([p["rougeL_f"] for p in subset]),
            f"{prefix}_lcs_ratio": _avg([p["lcs_ratio"] for p in subset]),
        }

    stats = {
        "n_probes": len(probes),
        "avg_rouge1_f": _avg([p["rouge1_f"] for p in probes]),
        "avg_rouge2_f": _avg([p["rouge2_f"] for p in probes]),
        "avg_rougeL_f": _avg([p["rougeL_f"] for p in probes]),
        "avg_lcs_ratio": _avg([p["lcs_ratio"] for p in probes]),
        **_stats(pre, "pre"),
        **_stats(post, "post"),
    }

    # Memorization signal heuristic
    pre_r = stats.get("pre_rougeL_f")
    post_r = stats.get("post_rougeL_f")
    if pre_r is not None and pre_r > 0.5:
        stats["signal"] = "strong"
    elif pre_r is not None and post_r is not None and (pre_r - post_r) > 0.10:
        stats["signal"] = "moderate"
    elif pre_r is not None and post_r is not None and (pre_r - post_r) > 0.05:
        stats["signal"] = "weak"
    elif pre_r is not None and pre_r > 0.3:
        stats["signal"] = "moderate"
    else:
        stats["signal"] = "none"

    return stats


# ── HTML generation ──────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blob = json.dumps(data)
    return (_HTML_TEMPLATE
            .replace("__DATA_PLACEHOLDER__", blob)
            .replace("__TIMESTAMP__", ts))


_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Leakage – Completion Probe</title>
<style>
  :root {
    --bg:#0d1117;--surface:#161b22;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;
    --green:#3fb950;--yellow:#d29922;--red:#f85149;--orange:#db6d28;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);padding:2rem;line-height:1.5}
  h1{font-size:1.6rem;margin-bottom:.3rem}
  h2{font-size:1.15rem;margin:1.8rem 0 .8rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
  .sub{color:var(--muted);margin-bottom:1.5rem;font-size:.95rem}
  .stats{display:flex;gap:1.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem 1.2rem;min-width:130px}
  .stat .num{font-size:1.8rem;font-weight:700}
  .stat .label{color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}
  .verdict{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.3rem;margin-bottom:1.5rem;font-size:.92rem;line-height:1.6}
  .verdict-warn{border-left:4px solid var(--yellow)} .verdict-ok{border-left:4px solid var(--green)}
  .verdict strong{color:var(--yellow)} .verdict-ok strong{color:var(--green)}
  .controls{display:flex;gap:.8rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center}
  .controls input,.controls select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.45rem .7rem;font-size:.9rem;outline:none}
  .controls input:focus,.controls select:focus{border-color:var(--accent)}
  .controls input{width:260px}
  table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:.86rem}
  th{background:#1c2128;text-align:left;padding:.55rem .7rem;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;cursor:pointer;user-select:none;white-space:nowrap}
  th:hover{color:var(--text)} th .arrow{margin-left:.3rem;font-size:.7rem}
  td{padding:.45rem .7rem;border-bottom:1px solid var(--border);vertical-align:middle}
  tr:last-child td{border-bottom:none} tr:hover td{background:#1c2128}
  .platform-tag{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.75rem;font-weight:600;background:#1c2128;border:1px solid var(--border)}
  .closed-tag{display:inline-block;padding:.05rem .35rem;border-radius:4px;font-size:.6rem;font-weight:700;background:#2a1215;color:var(--red);margin-left:.3rem;vertical-align:middle}
  .signal{display:inline-block;padding:.15rem .55rem;border-radius:10px;font-size:.72rem;font-weight:700;white-space:nowrap}
  .signal-none{background:#0d2818;color:var(--green)}
  .signal-weak{background:#1a1800;color:var(--yellow)}
  .signal-moderate{background:#291800;color:var(--orange)}
  .signal-strong{background:#2a1215;color:var(--red)}
  .num-cell{font-variant-numeric:tabular-nums;text-align:right}
  .bar-wrap{display:flex;align-items:center;gap:.4rem;min-width:120px}
  .bar-track{flex:1;height:7px;background:#21262d;border-radius:4px;overflow:hidden;min-width:60px}
  .bar-fill{height:100%;border-radius:4px}
  .bar-val{font-size:.78rem;font-variant-numeric:tabular-nums;min-width:36px;text-align:right}
  details{margin-bottom:.8rem}
  summary{cursor:pointer;padding:.5rem .7rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:.88rem;font-weight:600}
  summary:hover{background:#1c2128}
  details[open] summary{border-radius:6px 6px 0 0;border-bottom:none}
  details table{border-radius:0 0 8px 8px}
  .truncate{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block;vertical-align:middle}
  .pre-tag{color:var(--yellow);font-weight:600;font-size:.72rem}
  .post-tag{color:var(--green);font-weight:600;font-size:.72rem}
  @media(max-width:900px){body{padding:1rem}.stats{gap:.8rem}.stat{min-width:100px;padding:.6rem .8rem}.stat .num{font-size:1.4rem}}
</style>
</head>
<body>

<h1>Data Leakage – Abstract Completion Probe</h1>
<p class="sub">Generated by <code>scripts/check_data_leakage_completion.py</code> · __TIMESTAMP__</p>

<div class="stats" id="statsCards"></div>
<div id="verdictBox"></div>

<h2>Per-Model Summary</h2>
<p class="sub" style="margin-bottom:.8rem">For each model, ROUGE-L F1 scores are compared between articles published <span class="pre-tag">BEFORE</span> vs <span class="post-tag">AFTER</span> the model's training cutoff. A large gap (Δ) suggests memorization.</p>

<div class="controls">
  <input type="text" id="search" placeholder="Filter models…">
  <select id="signalFilter"><option value="">All signals</option>
    <option value="strong">🔴 Strong</option><option value="moderate">🟠 Moderate</option>
    <option value="weak">🟡 Weak</option><option value="none">🟢 None</option>
  </select>
</div>

<table id="modelTable">
<thead><tr>
  <th data-col="0">Platform <span class="arrow"></span></th>
  <th data-col="1">Model <span class="arrow"></span></th>
  <th data-col="2">Cutoff <span class="arrow"></span></th>
  <th data-col="3">Pre ROUGE-L <span class="arrow"></span></th>
  <th data-col="4">Post ROUGE-L <span class="arrow"></span></th>
  <th data-col="5">Δ <span class="arrow"></span></th>
  <th data-col="6">Pre ROUGE-2 <span class="arrow"></span></th>
  <th data-col="7">Post ROUGE-2 <span class="arrow"></span></th>
  <th data-col="8">Pre LCS% <span class="arrow"></span></th>
  <th data-col="9">Post LCS% <span class="arrow"></span></th>
  <th data-col="10">Signal <span class="arrow"></span></th>
</tr></thead>
<tbody id="modelBody"></tbody>
</table>

<h2>Detailed Probe Results</h2>
<div id="detailSection"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const CLOSED = new Set(["openai","anthropic","mistral"]);
const SIG_ORDER = {strong:0,moderate:1,weak:2,none:3};

// ── helpers ──
function fmt(v){return v===null||v===undefined?"—":(v*100).toFixed(1)+"%"}
function fmtDelta(a,b){if(a==null||b==null)return "—";const d=((a-b)*100).toFixed(1);return (d>=0?"+":"")+d+"%"}
function barColor(v){if(v==null)return "var(--muted)";if(v<0.15)return "var(--green)";if(v<0.30)return "var(--yellow)";if(v<0.50)return "var(--orange)";return "var(--red)"}
function sigClass(s){return "signal-"+(s||"none")}

// ── stats ──
const ms = DATA.model_summaries;
const nModels = ms.length;
const nProbes = ms.reduce((s,m)=>s+m.n_probes,0);
const withSignal = ms.filter(m=>m.signal!=="none").length;
const avgPre = ms.filter(m=>m.pre_rougeL_f!=null).reduce((s,m)=>({t:s.t+m.pre_rougeL_f,n:s.n+1}),{t:0,n:0});
const avgPost = ms.filter(m=>m.post_rougeL_f!=null).reduce((s,m)=>({t:s.t+m.post_rougeL_f,n:s.n+1}),{t:0,n:0});

document.getElementById("statsCards").innerHTML = `
  <div class="stat"><div class="num">${nModels}</div><div class="label">Models Tested</div></div>
  <div class="stat"><div class="num">${nProbes}</div><div class="label">Completion Probes</div></div>
  <div class="stat"><div class="num">${DATA.sample_articles.length}</div><div class="label">Sample Articles</div></div>
  <div class="stat"><div class="num" style="color:var(--yellow)">${avgPre.n?((avgPre.t/avgPre.n)*100).toFixed(1)+"%":"—"}</div><div class="label">Avg Pre-cutoff ROUGE-L</div></div>
  <div class="stat"><div class="num" style="color:var(--green)">${avgPost.n?((avgPost.t/avgPost.n)*100).toFixed(1)+"%":"—"}</div><div class="label">Avg Post-cutoff ROUGE-L</div></div>
  <div class="stat"><div class="num" style="color:${withSignal?"var(--orange)":"var(--green)"}">${withSignal}</div><div class="label">Models w/ Signal</div></div>
`;

// ── verdict ──
const v = document.getElementById("verdictBox");
if(withSignal>0){
  v.innerHTML=`<div class="verdict verdict-warn">⚠️ <strong>${withSignal} model(s) show potential memorization signal.</strong>
  Models with higher ROUGE scores on pre-cutoff articles may have seen those abstracts during training.
  This warrants further investigation (e.g., larger sample sizes, performance-by-date analysis).</div>`;
}else{
  v.innerHTML=`<div class="verdict verdict-ok">✅ <strong>No memorization signal detected.</strong>
  ROUGE scores on pre-cutoff and post-cutoff articles are comparable across all tested models.
  This suggests models are not reproducing memorized abstracts.</div>`;
}

// ── model summary table ──
const tbody = document.getElementById("modelBody");
ms.forEach(m=>{
  const isClosed = CLOSED.has(m.platform);
  const delta = (m.pre_rougeL_f!=null&&m.post_rougeL_f!=null)?(m.pre_rougeL_f-m.post_rougeL_f):null;
  const tr = document.createElement("tr");
  tr.dataset.signal = m.signal;
  tr.innerHTML = `
    <td data-sort="${m.platform}"><span class="platform-tag">${m.platform}</span></td>
    <td data-sort="${m.model}">${m.model}${isClosed?'<span class="closed-tag">CLOSED</span>':''}</td>
    <td data-sort="${m.training_cutoff||'9999'}" class="num-cell">${m.training_cutoff||'—'}</td>
    <td data-sort="${m.pre_rougeL_f!=null?m.pre_rougeL_f.toFixed(4):'9'}">
      <div class="bar-wrap"><span class="bar-val" style="color:${barColor(m.pre_rougeL_f)}">${fmt(m.pre_rougeL_f)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(m.pre_rougeL_f||0)*100}%;background:${barColor(m.pre_rougeL_f)}"></div></div></div></td>
    <td data-sort="${m.post_rougeL_f!=null?m.post_rougeL_f.toFixed(4):'9'}">
      <div class="bar-wrap"><span class="bar-val" style="color:${barColor(m.post_rougeL_f)}">${fmt(m.post_rougeL_f)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(m.post_rougeL_f||0)*100}%;background:${barColor(m.post_rougeL_f)}"></div></div></div></td>
    <td data-sort="${delta!=null?delta.toFixed(4):'0'}" class="num-cell" style="color:${delta!=null&&delta>0.05?'var(--orange)':'var(--muted)'};font-weight:600">${fmtDelta(m.pre_rougeL_f,m.post_rougeL_f)}</td>
    <td data-sort="${m.pre_rouge2_f!=null?m.pre_rouge2_f.toFixed(4):'9'}" class="num-cell">${fmt(m.pre_rouge2_f)}</td>
    <td data-sort="${m.post_rouge2_f!=null?m.post_rouge2_f.toFixed(4):'9'}" class="num-cell">${fmt(m.post_rouge2_f)}</td>
    <td data-sort="${m.pre_lcs_ratio!=null?m.pre_lcs_ratio.toFixed(4):'9'}" class="num-cell">${fmt(m.pre_lcs_ratio)}</td>
    <td data-sort="${m.post_lcs_ratio!=null?m.post_lcs_ratio.toFixed(4):'9'}" class="num-cell">${fmt(m.post_lcs_ratio)}</td>
    <td data-sort="${SIG_ORDER[m.signal]!=null?SIG_ORDER[m.signal]:9}"><span class="signal ${sigClass(m.signal)}">${m.signal}</span></td>
  `;
  tbody.appendChild(tr);
});

// ── filtering ──
const allModelRows=[...tbody.querySelectorAll("tr")];
const searchEl=document.getElementById("search");
const sigFilt=document.getElementById("signalFilter");
function filterModels(){
  const q=searchEl.value.toLowerCase();const sig=sigFilt.value;
  allModelRows.forEach(r=>{
    const text=r.textContent.toLowerCase();
    r.style.display=(text.includes(q)&&(!sig||r.dataset.signal===sig))?"":"none";
  });
}
searchEl.addEventListener("input",filterModels);
sigFilt.addEventListener("change",filterModels);

// ── sorting ──
let sortCol=-1,sortAsc=true;
document.querySelectorAll("#modelTable th[data-col]").forEach(th=>{
  th.addEventListener("click",()=>{
    const col=+th.dataset.col;
    if(sortCol===col)sortAsc=!sortAsc;else{sortCol=col;sortAsc=true}
    document.querySelectorAll("#modelTable th .arrow").forEach(a=>a.textContent="");
    th.querySelector(".arrow").textContent=sortAsc?"▲":"▼";
    allModelRows.sort((a,b)=>{
      const av=a.children[col].dataset.sort||a.children[col].textContent.trim();
      const bv=b.children[col].dataset.sort||b.children[col].textContent.trim();
      return sortAsc?av.localeCompare(bv,undefined,{numeric:true}):bv.localeCompare(av,undefined,{numeric:true});
    });
    allModelRows.forEach(r=>tbody.appendChild(r));
  });
});

// ── detailed per-model probe results ──
const detailSection=document.getElementById("detailSection");
const probesByModel={};
DATA.probe_results.forEach(p=>{(probesByModel[p.model]=probesByModel[p.model]||[]).push(p)});

ms.forEach(m=>{
  const probes=probesByModel[m.model]||[];
  if(!probes.length)return;
  const det=document.createElement("details");
  const isClosed=CLOSED.has(m.platform);
  det.innerHTML=`<summary>${m.model}${isClosed?' <span class="closed-tag">CLOSED</span>':''} — ${probes.length} probes, signal: <span class="signal ${sigClass(m.signal)}">${m.signal}</span></summary>
  <table><thead><tr>
    <th>Journal</th><th>Pub Date</th><th>Class</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th><th>LCS%</th><th style="min-width:200px">First Sentence</th>
  </tr></thead><tbody>${probes.map(p=>`<tr>
    <td>${p.journal_id}</td>
    <td class="num-cell">${p.publication_date}</td>
    <td><span class="${p.temporal_class==='pre_cutoff'?'pre-tag':'post-tag'}">${p.temporal_class.replace('_',' ')}</span></td>
    <td class="num-cell">${fmt(p.rouge1_f)}</td>
    <td class="num-cell">${fmt(p.rouge2_f)}</td>
    <td class="num-cell" style="color:${barColor(p.rougeL_f)};font-weight:600">${fmt(p.rougeL_f)}</td>
    <td class="num-cell">${fmt(p.lcs_ratio)}</td>
    <td><span class="truncate" title="${p.first_sentence.replace(/"/g,'&quot;')}">${p.first_sentence}</span></td>
  </tr>`).join('')}</tbody></table>`;
  detailSection.appendChild(det);
});
</script>
</body>
</html>'''


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # NLTK tokeniser
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # ── load data ──
    log.info("Loading data …")
    with open(LEAKAGE_JSON) as f:
        leakage = json.load(f)
    with open(DATASET_JSON) as f:
        articles = json.load(f)

    # ── select at-risk models ──
    at_risk = [
        m for m in leakage["models"]
        if m["platform"] in TESTABLE_PLATFORMS
        and m["pct_potentially_leaked"] is not None
        and m["pct_potentially_leaked"] > MIN_OVERLAP_PCT
    ]
    if not at_risk:
        log.warning("No at-risk models found.  Nothing to do.")
        return

    log.info(f"At-risk models to test: {len(at_risk)}")
    for m in at_risk:
        closed = " [CLOSED]" if m["platform"] in CLOSED_PLATFORMS else ""
        log.info(f"  {m['platform']:>10} | {m['model']:<40} | "
                 f"overlap {m['pct_potentially_leaked']}%{closed}")

    # ── select sample articles ──
    samples = select_samples(articles)
    log.info(f"Selected {len(samples)} sample articles from "
             f"{len({s['journal_id'] for s in samples})} journals")

    # ── load cache ──
    cache = load_cache()

    # ── run probes ──
    all_probes: list[dict] = []
    total = len(at_risk) * len(samples)
    done = 0

    for model_info in at_risk:
        platform = model_info["platform"]
        model    = model_info["model"]
        cutoff   = model_info.get("training_cutoff")
        cutoff_dt = parse_partial_date(cutoff)

        log.info(f"\n{'═' * 60}")
        log.info(f"  {model}  ({platform}, cutoff {cutoff})")
        log.info(f"{'═' * 60}")

        for sample in samples:
            done += 1
            art_id    = sample["article"]["id"]
            first     = sample["first_sentence"]
            remaining = sample["remaining_text"]

            if not remaining.strip():
                log.warning(f"  [{done}/{total}] skip — no text after first sentence")
                continue

            ckey = _cache_key(model, art_id)
            prompt = USER_PROMPT_TEMPLATE.format(first_sentence=first)

            # cache lookup
            if ckey in cache:
                completion = cache[ckey]["completion"]
                log.info(f"  [{done}/{total}] cache hit  {art_id[:70]}")
            else:
                try:
                    log.info(f"  [{done}/{total}] querying   {art_id[:70]}")
                    completion = query_model(platform, model, prompt)
                    cache[ckey] = {
                        "completion": completion,
                        "model": model,
                        "article_id": art_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    save_cache(cache)       # persist after every call
                    time.sleep(REQUEST_DELAY)
                except Exception as exc:
                    log.error(f"  [{done}/{total}] ERROR — {exc}")
                    continue

            cleaned = clean_completion(completion, first)
            scores  = score_completion(remaining, cleaned)

            pub_dt = parse_partial_date(sample["pub_date"])
            if pub_dt and cutoff_dt:
                t_class = "pre_cutoff" if pub_dt <= cutoff_dt else "post_cutoff"
            else:
                t_class = "unknown"

            all_probes.append({
                "model": model,
                "platform": platform,
                "training_cutoff": cutoff,
                "article_id": art_id,
                "article_title": sample["article"].get("title", ""),
                "journal_id": sample["journal_id"],
                "publication_date": sample["pub_date"],
                "sample_group": sample["sample_group"],
                "temporal_class": t_class,
                "first_sentence": first,
                "reference_snippet": remaining[:400],
                "completion_snippet": cleaned[:400],
                **scores,
            })

    save_cache(cache)

    # ── aggregate ──
    model_summaries: list[dict] = []
    probes_by_model: dict[str, list[dict]] = defaultdict(list)
    for p in all_probes:
        probes_by_model[p["model"]].append(p)

    for mi in at_risk:
        model = mi["model"]
        probes = probes_by_model.get(model, [])
        stats = aggregate_model(probes)
        model_summaries.append({
            "model": model,
            "platform": mi["platform"],
            "training_cutoff": mi.get("training_cutoff"),
            "is_closed_source": mi["platform"] in CLOSED_PLATFORMS,
            "pct_overlap": mi["pct_potentially_leaked"],
            **stats,
        })

    # sort: strongest signal first
    sig_order = {"strong": 0, "moderate": 1, "weak": 2, "none": 3}
    model_summaries.sort(key=lambda m: sig_order.get(m.get("signal", "none"), 9))

    # ── build output ──
    output = {
        "description": (
            "Abstract completion probe: models are prompted with the first "
            "sentence of sample abstracts and asked to continue. "
            "High ROUGE overlap with the actual continuation suggests memorization."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "min_overlap_pct": MIN_OVERLAP_PCT,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
        },
        "sample_articles": [
            {
                "article_id": s["article"]["id"],
                "title": s["article"].get("title", ""),
                "journal_id": s["journal_id"],
                "publication_date": s["pub_date"],
                "sample_group": s["sample_group"],
                "first_sentence": s["first_sentence"],
            }
            for s in samples
        ],
        "model_summaries": model_summaries,
        "probe_results": all_probes,
    }

    # ── write JSON ──
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"JSON → {OUTPUT_JSON.relative_to(ROOT)}")

    # ── write HTML ──
    with open(OUTPUT_HTML, "w") as f:
        f.write(generate_html(output))
    log.info(f"HTML → {OUTPUT_HTML.relative_to(ROOT)}")

    # ── console summary ──
    print()
    print("=" * 80)
    print("ABSTRACT COMPLETION PROBE — SUMMARY")
    print("=" * 80)
    print(f"\n  Models tested   : {len(model_summaries)}")
    print(f"  Articles probed : {len(samples)}")
    print(f"  Total probes    : {len(all_probes)}")
    print()
    print("-" * 100)
    print(f"  {'Model':<40} {'Cutoff':>7}  {'Pre RL':>7}  {'Post RL':>8}  "
          f"{'Δ':>7}  {'Pre R2':>7}  {'Post R2':>8}  Signal")
    print("-" * 100)
    for m in model_summaries:
        closed = " [C]" if m["is_closed_source"] else ""
        name = f"{m['model']}{closed}"
        if len(name) > 40:
            name = name[:37] + "..."
        pre_rl  = f"{m['pre_rougeL_f']:.3f}" if m.get("pre_rougeL_f") is not None else "  —  "
        post_rl = f"{m['post_rougeL_f']:.3f}" if m.get("post_rougeL_f") is not None else "  —  "
        if m.get("pre_rougeL_f") is not None and m.get("post_rougeL_f") is not None:
            delta = f"{m['pre_rougeL_f'] - m['post_rougeL_f']:+.3f}"
        else:
            delta = "  —  "
        pre_r2  = f"{m['pre_rouge2_f']:.3f}" if m.get("pre_rouge2_f") is not None else "  —  "
        post_r2 = f"{m['post_rouge2_f']:.3f}" if m.get("post_rouge2_f") is not None else "  —  "
        print(f"  {name:<40} {m.get('training_cutoff','—'):>7}  {pre_rl:>7}  "
              f"{post_rl:>8}  {delta:>7}  {pre_r2:>7}  {post_r2:>8}  {m.get('signal','—')}")
    print("-" * 100)

    signals = [m for m in model_summaries if m.get("signal", "none") != "none"]
    if signals:
        print(f"\n  ⚠  {len(signals)} model(s) show memorization signal:")
        for m in signals:
            print(f"     • {m['model']} — {m['signal']}")
    else:
        print("\n  ✅  No memorization signal detected across any tested model.")

    print(f"\n  Outputs:")
    print(f"    JSON → {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"    HTML → {OUTPUT_HTML.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
