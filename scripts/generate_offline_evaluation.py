#!/usr/bin/env python3
"""Generate a self-contained offline evaluation HTML file.

Creates an HTML file that reviewers can open in any browser to evaluate
summaries without needing access to the evaluation server. Progress is
saved in the browser's localStorage and can be exported as JSON.

Usage
-----
    python generate_offline_evaluation.py \\
        openai_gpt-4o anthropic_claude-opus-4-20250514 \\
        local:textrank ollama_gemma3:270M \\
        -o evaluation_alice.html

    # Simple single-score mode:
    python generate_offline_evaluation.py \\
        openai_gpt-4o local:textrank ... --simple-ratings \\
        -o eval_simple.html

    # Side-by-side ranking mode:
    python generate_offline_evaluation.py \\
        openai_gpt-4o local:textrank ... --side-by-side \\
        -o eval_ranking.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

# Import shared data-loading logic from the evaluation server
sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_evaluation_server import (
    DEFAULT_GOLDSTANDARD,
    DEFAULT_NUM_PAPERS,
    DEFAULT_RESULTS,
    DETAILED_CRITERIA,
    SIMPLE_CRITERIA,
    load_evaluation_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Assignment generation ─────────────────────────────────────────────────


def generate_assignments(
    eval_data: dict, rating_mode: str, seed: str | None = None
) -> list[dict]:
    """Create a shuffled assignment list (same logic as the server)."""
    rng = random.Random(seed)

    if rating_mode == "side-by-side":
        labels = ["A", "B", "C", "D"]
        assignments: list[dict] = []
        for pi in range(len(eval_data["papers"])):
            shuffled = list(labels)
            rng.shuffle(shuffled)
            assignments.append(
                {
                    "paper_index": pi,
                    "label_map": dict(zip(shuffled, eval_data["models"])),
                }
            )
        rng.shuffle(assignments)
    else:
        assignments = [
            {"paper_index": pi, "model": model}
            for pi in range(len(eval_data["papers"]))
            for model in eval_data["models"]
        ]
        rng.shuffle(assignments)

    return assignments


# ── HTML template ─────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Offline Evaluation &ndash; Text Summarization Benchmark</title>
<script>document.documentElement.dataset.theme=localStorage.getItem("theme")||"light"</script>
<style>
:root{--bg:#ffffff;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
--muted:#656d76;--accent:#0969da;--green:#1a7f37;--red:#cf222e;
--yellow:#9a6700;--radius:8px;--btn-text:#ffffff;
--label-a:#0969da;--label-b:#1a7f37;--label-c:#9a6700;--label-d:#8250df}
[data-theme="dark"]{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;
--yellow:#d29922;--btn-text:#0d1117;
--label-a:#58a6ff;--label-b:#3fb950;--label-c:#d29922;--label-d:#bc8cff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}
.container{max-width:800px;margin:0 auto;padding:16px 16px 64px}
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
.criterion-header{display:flex;align-items:baseline;gap:6px;margin-bottom:4px;flex-wrap:wrap}
.criterion-label{font-weight:600;font-size:0.88rem}
.criterion-desc{font-size:0.78rem;color:var(--muted)}
.likert{display:flex;gap:4px}
.likert label{flex:1;text-align:center;padding:6px 2px;
border:2px solid var(--border);border-radius:6px;cursor:pointer;
font-size:0.8rem;transition:all 0.15s;user-select:none;line-height:1.2}
.likert label:hover{border-color:var(--accent);background:rgba(88,166,255,0.08)}
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
.toolbar{display:flex;align-items:center;justify-content:space-between;
margin-bottom:14px;padding:10px 14px;background:var(--surface);
border:1px solid var(--border);border-radius:var(--radius);font-size:0.82rem;
flex-wrap:wrap;gap:8px}
.toolbar .name-display{font-weight:600;color:var(--text)}
.toolbar .btn-group{display:flex;gap:6px}
.tool-btn{background:none;border:1px solid var(--border);
color:var(--muted);padding:4px 10px;font-size:0.78rem;border-radius:4px;
cursor:pointer;margin:0;font-weight:500}
.tool-btn:hover{border-color:var(--accent);color:var(--accent);background:transparent}
.tool-btn.primary{border-color:var(--green);color:var(--green)}
.tool-btn.primary:hover{background:var(--green);color:var(--btn-text)}
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
vertical-align:middle;line-height:1;font-style:italic;font-family:Georgia,serif;padding:0}
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
.nav-legend{display:flex;gap:10px;font-size:0.68rem;color:var(--muted);margin-bottom:6px;padding-left:6px}
.nav-legend-item{display:flex;align-items:center;gap:3px}
.nav-swatch{width:8px;height:8px;border-radius:2px;border:1.5px solid var(--border)}
.nav-swatch.sw-done{background:var(--green);border-color:var(--green)}
.nav-swatch.sw-current{border-color:var(--accent);box-shadow:0 0 0 1px rgba(88,166,255,0.3)}
.nav-swatch.sw-pending{background:var(--bg)}
.status-badge{font-size:0.7rem;text-transform:none;letter-spacing:0;
padding:2px 8px;border-radius:4px;margin-left:8px;font-weight:600}
.status-badge.completed{color:var(--green);background:rgba(26,127,55,0.1)}
/* Welcome screen */
.welcome-center{display:flex;align-items:center;justify-content:center;min-height:80vh}
.welcome-box{max-width:480px;width:100%;padding:24px}
.welcome-box h1{font-size:1.4rem;font-weight:600;margin-bottom:4px}
.welcome-box .subtitle{color:var(--muted);margin-bottom:32px;font-size:0.95rem}
input[type=text]{width:100%;padding:10px 14px;background:var(--bg);
border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:1rem;outline:none}
input[type=text]:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(88,166,255,0.15)}
input[type=text]::placeholder{color:var(--muted)}
.start-btn{display:block;width:100%;padding:10px 20px;margin-top:12px}
.error{color:var(--red);font-size:0.85rem;margin-top:8px;display:none}
.divider{text-align:center;color:var(--muted);margin:20px 0;font-size:0.85rem}
.info-text{font-size:0.85rem;color:var(--muted);margin-top:24px;text-align:center}
/* Side-by-side specific */
.summary-card{background:var(--surface);border:2px solid var(--border);
border-radius:var(--radius);padding:20px;margin-bottom:16px;transition:border-color 0.2s}
.summary-card.ranked{border-color:var(--accent)}
.summary-label{display:inline-flex;align-items:center;gap:8px;
font-weight:700;font-size:1rem;margin-bottom:10px}
.summary-label .letter{display:inline-flex;align-items:center;
justify-content:center;width:28px;height:28px;border-radius:6px;
font-size:0.85rem;font-weight:700;color:var(--btn-text)}
.letter-a{background:var(--label-a)}.letter-b{background:var(--label-b)}
.letter-c{background:var(--label-c)}.letter-d{background:var(--label-d)}
.summary-text{background:var(--bg);border:1px solid var(--border);
border-radius:6px;padding:14px;font-size:0.92rem;line-height:1.7;
color:var(--text);white-space:pre-wrap;margin-bottom:12px}
.rank-row{display:flex;align-items:center;gap:10px}
.rank-label{font-size:0.82rem;color:var(--muted);font-weight:600;min-width:40px}
.rank-buttons{display:flex;gap:6px}
.rank-buttons label{text-align:center;padding:8px 14px;
border:2px solid var(--border);border-radius:8px;cursor:pointer;
font-size:0.85rem;transition:all 0.15s;user-select:none;
line-height:1.3;min-width:56px}
.rank-buttons label:hover{border-color:var(--accent);background:rgba(88,166,255,0.08)}
.rank-buttons label.selected{border-color:var(--accent);
background:var(--accent);color:var(--btn-text);font-weight:600}
.rank-buttons label .anchor{display:block;font-size:0.68rem;margin-top:2px;opacity:0.8}
.rank-badge{display:none;font-size:0.78rem;padding:2px 8px;border-radius:4px;
background:var(--accent);color:var(--btn-text);font-weight:600;margin-left:auto}
.summary-card.ranked .rank-badge{display:inline-block}
.hint{font-size:0.82rem;color:var(--muted);margin-bottom:16px;font-style:italic}
textarea{width:100%;min-height:60px;padding:8px 10px;background:var(--bg);
border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:0.85rem;font-family:inherit;resize:vertical;outline:none}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
</style>
</head>
<body>
<div class="modal-overlay" id="info-modal" onclick="if(event.target===this)closeInfo()">
<div class="modal">
<h3>Evaluation Criteria &mdash; SummEval Framework</h3>
<p>These four dimensions come from the SummEval framework
(Fabbri&nbsp;et&nbsp;al.,&nbsp;2021), the <em>de&nbsp;facto</em> standard for
human evaluation of text summarization. Each is rated on a 1&ndash;5 Likert scale.</p>
<h4>Coherence</h4>
<p>The summary should be well-structured and well-organized. It should not just
be a heap of related information, but should build from sentence to sentence to
a coherent body of information about a topic.</p>
<h4>Consistency</h4>
<p>The factual alignment between the summary and the summarized source document.
A factually consistent summary contains only statements that are entailed by the
source document. Penalize summaries that contain hallucinated facts.</p>
<h4>Fluency</h4>
<p>The quality of individual sentences. They should have no formatting problems,
capitalization errors or obviously ungrammatical sentences (e.g., fragments,
missing components) that make the text difficult to read.</p>
<h4>Relevance</h4>
<p>Selection of important content from the source. The summary should include
only important information from the source document. Penalize summaries which
contain redundancies and excess information.</p>
<p class="ref">Fabbri, A.&nbsp;R. et&nbsp;al. (2021). SummEval: Re-evaluating
Summarization Evaluation. <em>Transactions of the Association for Computational
Linguistics</em>, 9, 391&ndash;409.</p>
<button class="close-btn" onclick="closeInfo()">Close</button>
</div>
</div>
<button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
<div id="app"></div>
<script type="application/json" id="eval-data">__DATA_JSON__</script>
<script>
// ── Config & State ───────────────────────────────────────────────────────
var CONFIG=JSON.parse(document.getElementById('eval-data').textContent);
var STORAGE_KEY='offline_eval_'+CONFIG.data_hash;
var state=loadState();
var sideRankings={};

function loadState(){
  try{var raw=localStorage.getItem(STORAGE_KEY);
    if(raw){var s=JSON.parse(raw);if(s&&s.assessments)return s;}}catch(e){}
  return{reviewer_name:null,started_at:null,current_index:0,assessments:[]};
}
function saveState(){
  try{localStorage.setItem(STORAGE_KEY,JSON.stringify(state));}catch(e){}
}

// ── Theme ────────────────────────────────────────────────────────────────
function toggleTheme(){
  var t=document.documentElement.dataset.theme==='dark'?'light':'dark';
  document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);
}

// ── Utilities ────────────────────────────────────────────────────────────
function escH(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function nowISO(){return new Date().toISOString();}

// ── Modal ────────────────────────────────────────────────────────────────
function openInfo(){document.getElementById('info-modal').classList.add('open');}
function closeInfo(){document.getElementById('info-modal').classList.remove('open');}

// ── Render router ────────────────────────────────────────────────────────
function render(){
  if(!state.reviewer_name){renderWelcome();return;}
  var completed=state.assessments.length;
  var total=CONFIG.assignments.length;
  if(state.current_index>=total&&completed>=total){renderDone();}
  else{renderAssessment(state.current_index);}
}

// ── Welcome view ─────────────────────────────────────────────────────────
function renderWelcome(){
  var n=CONFIG.papers.length;var m=CONFIG.models.length;
  var info;
  if(CONFIG.rating_mode==='side-by-side'){
    info=n+' papers \u2014 rank '+m+' summaries per paper ('+n+' ranking tasks)';
  }else{
    info=n+' papers \u00d7 '+m+' models = '+(n*m)+' assessments';
  }
  document.getElementById('app').innerHTML=
    '<div class="welcome-center"><div class="welcome-box">'+
    '<h1>Human Evaluation</h1>'+
    '<p class="subtitle">Text Summarization Benchmark (Offline)</p>'+
    '<div class="card"><h2 style="font-size:1rem;margin-bottom:12px;color:var(--accent)">'+
    'Start evaluation</h2>'+
    '<input type="text" id="name-input" placeholder="Your name" '+
    'onkeydown="if(event.key===\'Enter\')startEvaluation()">'+
    '<div class="error" id="name-error"></div>'+
    '<button class="start-btn" onclick="startEvaluation()">Start</button></div>'+
    '<div class="divider">\u2014 or \u2014</div>'+
    '<div class="card"><h2 style="font-size:1rem;margin-bottom:12px;color:var(--accent)">'+
    'Resume from export</h2>'+
    '<p style="font-size:0.85rem;color:var(--muted);margin-bottom:8px">'+
    'Import a previously exported JSON file to continue.</p>'+
    '<input type="file" id="import-input" accept=".json" '+
    'onchange="importFromFile(this.files[0])" style="font-size:0.85rem">'+
    '<div class="error" id="import-error"></div></div>'+
    '<p class="info-text">'+escH(info)+'</p></div></div>';
}

function startEvaluation(){
  var name=document.getElementById('name-input').value.trim();
  var err=document.getElementById('name-error');
  err.style.display='none';
  if(!name){err.textContent='Please enter your name.';err.style.display='block';return;}
  if(name.length>200){err.textContent='Name too long.';err.style.display='block';return;}
  state.reviewer_name=name;
  state.started_at=nowISO();
  saveState();
  render();
}

// ── Assessment view ──────────────────────────────────────────────────────
function renderAssessment(idx){
  var completed=state.assessments.length;
  var total=CONFIG.assignments.length;
  if(idx<0)idx=0;
  if(idx>completed)idx=completed;
  if(idx>=total){renderDone();return;}
  state.current_index=idx;
  saveState();

  var assignment=CONFIG.assignments[idx];
  var paper=CONFIG.papers[assignment.paper_index];
  var isSubmitted=idx<completed;
  var previous=isSubmitted?state.assessments[idx]:null;

  var pct=total>0?Math.round(completed/total*100):0;
  var navHtml=buildNavGrid(idx,completed,total);
  var toolbarHtml=
    '<div class="toolbar"><span>Reviewer: <strong class="name-display">'+
    escH(state.reviewer_name)+'</strong></span><div class="btn-group">'+
    '<button class="tool-btn primary" onclick="exportResults()" title="Download results JSON">'+
    '\u2913 Export JSON</button>'+
    '<label class="tool-btn" style="cursor:pointer" title="Import previous export">'+
    '\u2912 Import<input type="file" accept=".json" onchange="importFromFile(this.files[0])" '+
    'style="display:none"></label></div></div>';
  var headerHtml=
    '<header><h1>Text Summarization Evaluation</h1>'+toolbarHtml+
    '<div class="progress-wrap"><div class="progress-bar" style="width:'+pct+'%"></div>'+
    '<div class="progress-text">'+completed+' / '+total+' completed ('+pct+'%)</div></div>'+
    navHtml+
    '<div class="nav-legend">'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-done"></span>Completed</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-current"></span>Current</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-pending"></span>Pending</span>'+
    '</div></header>';

  var mainHtml;
  if(CONFIG.rating_mode==='side-by-side'){
    mainHtml=renderSideBySideBody(idx,assignment,paper,isSubmitted,previous,total);
  }else{
    mainHtml=renderRatingBody(idx,assignment,paper,isSubmitted,previous,total);
  }

  document.getElementById('app').innerHTML=
    '<div class="container">'+headerHtml+'<main>'+mainHtml+'</main></div>';

  // Restore previous selections
  if(previous&&CONFIG.rating_mode!=='side-by-side'&&previous.ratings){
    for(var key in previous.ratings){
      var el=document.querySelector('.likert label[data-key="'+key+'"][data-val="'+previous.ratings[key]+'"]');
      if(el)el.classList.add('selected');
    }
  }
  if(previous&&CONFIG.rating_mode==='side-by-side'&&previous.rankings){
    sideRankings={};
    for(var label in previous.rankings){
      sideRankings[label]=previous.rankings[label];
      var card=document.getElementById('card-'+label);
      if(card){
        var el=card.querySelector('.rank-buttons label[data-rank="'+previous.rankings[label]+'"]');
        if(el){el.classList.add('selected');card.classList.add('ranked');
          document.getElementById('badge-'+label).textContent='#'+previous.rankings[label];}
      }
    }
  }else{
    sideRankings={};
  }
  if(previous&&previous.comment){
    var ta=document.getElementById('comment');
    if(ta)ta.value=previous.comment;
  }
  checkSubmittable();
}

function renderRatingBody(idx,assignment,paper,isSubmitted,previous,total){
  var bullets=paper.reference_highlights.map(function(h){return '<li>'+escH(h)+'</li>';}).join('');
  var summary=paper.summaries[assignment.model];
  var btnLabel=isSubmitted?'Update':'Submit';
  var statusBadge=isSubmitted?'<span class="status-badge completed">\u2714 completed</span>':'';

  var criteriaHTML='';
  for(var i=0;i<CONFIG.criteria.length;i++){
    var c=CONFIG.criteria[i];
    criteriaHTML+='<div class="criterion"><div class="criterion-header">'+
      '<span class="criterion-label">'+escH(c.label)+'</span>'+
      '<span class="criterion-desc">'+escH(c.description)+'</span></div><div class="likert">';
    for(var v=1;v<=5;v++){
      var anchor=(c.anchors&&c.anchors[String(v)])||'';
      criteriaHTML+='<label data-key="'+c.key+'" data-val="'+v+'" onclick="selectRating(this)">'+
        v+(anchor?'<span class="anchor">'+escH(anchor)+'</span>':'')+'</label>';
    }
    criteriaHTML+='</div></div>';
  }

  var infoBtnHtml=CONFIG.rating_mode==='detailed'?
    '<button class="info-btn" onclick="openInfo()" title="About these criteria">i</button>':'';

  return '<div class="card"><div class="card-header">Assessment '+(idx+1)+' of '+total+
    statusBadge+'</div><ul class="highlights">'+bullets+'</ul></div>'+
    '<div class="card"><div class="card-header">Generated Summary</div>'+
    '<div class="gen-text">'+escH(summary)+'</div></div>'+
    '<div class="card"><div class="card-header">Your Assessment '+infoBtnHtml+'</div>'+
    criteriaHTML+
    '<div style="margin-top:8px"><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:4px">'+
    'Comment (optional)</label>'+
    '<textarea id="comment" placeholder="Any observations\u2026"></textarea></div>'+
    '<div class="submit-row"><button id="submit-btn" onclick="submitAssessment()" disabled>'+
    btnLabel+'</button></div></div>';
}

function renderSideBySideBody(idx,assignment,paper,isSubmitted,previous,total){
  var bullets=paper.reference_highlights.map(function(h){return '<li>'+escH(h)+'</li>';}).join('');
  var labelMap=assignment.label_map;
  var labels=Object.keys(labelMap).sort();
  var colors={A:'a',B:'b',C:'c',D:'d'};
  var btnLabel=isSubmitted?'Update Rankings':'Submit Rankings';
  var statusBadge=isSubmitted?'<span class="status-badge completed">\u2714 completed</span>':'';

  var html='<div class="card"><div class="card-header">Paper '+(idx+1)+' of '+total+
    statusBadge+'</div><ul class="highlights">'+bullets+'</ul></div>'+
    '<p class="hint">Read all four summaries below, then rank them from 1 (best) to 4 (worst).</p>';

  for(var li=0;li<labels.length;li++){
    var label=labels[li];
    var model=labelMap[label];
    html+='<div class="summary-card" id="card-'+label+'">'+
      '<div class="summary-label"><span class="letter letter-'+colors[label]+'">'+label+'</span>'+
      ' Summary '+label+'<span class="rank-badge" id="badge-'+label+'"></span></div>'+
      '<div class="summary-text">'+escH(paper.summaries[model])+'</div>'+
      '<div class="rank-row"><span class="rank-label">Rank:</span><div class="rank-buttons">';
    for(var r=1;r<=4;r++){
      var anchor=r===1?'Best':(r===4?'Worst':'');
      html+='<label data-label="'+label+'" data-rank="'+r+'" onclick="selectRank(this)">'+
        r+(anchor?'<span class="anchor">'+anchor+'</span>':'')+'</label>';
    }
    html+='</div></div></div>';
  }

  html+='<div class="card"><div style="margin-bottom:12px">'+
    '<label style="font-size:0.85rem;color:var(--muted);display:block;margin-bottom:6px">'+
    'Comment (optional)</label>'+
    '<textarea id="comment" placeholder="Any observations\u2026"></textarea></div>'+
    '<div class="submit-row"><button id="submit-btn" onclick="submitAssessment()" disabled>'+
    btnLabel+'</button></div></div>';

  return html;
}

// ── Navigation grid ──────────────────────────────────────────────────────
function buildNavGrid(activeIdx,completed,total){
  var html='<div class="nav-grid">';
  for(var i=0;i<total;i++){
    var done=i<completed;
    var active=i===activeIdx;
    var reachable=i<=completed;
    var cls='nav-item';
    if(done)cls+=' done';
    if(active)cls+=' active';
    if(!reachable&&!done)cls+=' locked';
    var assignment=CONFIG.assignments[i];
    var paper=CONFIG.papers[assignment.paper_index];
    var tip=escH(paper.title||'').substring(0,60);
    if(assignment.model)tip+=' \u2014 '+escH(assignment.model);
    html+='<div class="'+cls+'"'+(reachable?' onclick="goTo('+i+')"':'')+
      ' title="'+tip+'">'+(i+1)+'</div>';
  }
  html+='</div>';
  return html;
}

function goTo(index){
  sideRankings={};
  renderAssessment(index);
}

// ── Rating interactions ──────────────────────────────────────────────────
function selectRating(el){
  var key=el.dataset.key;
  el.parentElement.querySelectorAll('label').forEach(function(l){l.classList.remove('selected');});
  el.classList.add('selected');
  checkSubmittable();
}

function selectRank(el){
  var label=el.dataset.label;
  var rank=parseInt(el.dataset.rank);
  // Deselect same rank from other labels
  var keys=Object.keys(sideRankings);
  for(var i=0;i<keys.length;i++){
    if(sideRankings[keys[i]]===rank&&keys[i]!==label){
      var prevLabel=keys[i];
      delete sideRankings[prevLabel];
      var prevCard=document.getElementById('card-'+prevLabel);
      if(prevCard){
        prevCard.classList.remove('ranked');
        prevCard.querySelectorAll('.rank-buttons label').forEach(function(l){l.classList.remove('selected');});
        document.getElementById('badge-'+prevLabel).textContent='';
      }
    }
  }
  sideRankings[label]=rank;
  var card=document.getElementById('card-'+label);
  card.querySelectorAll('.rank-buttons label').forEach(function(l){l.classList.remove('selected');});
  el.classList.add('selected');
  card.classList.add('ranked');
  document.getElementById('badge-'+label).textContent='#'+rank;
  checkSubmittable();
}

function checkSubmittable(){
  var btn=document.getElementById('submit-btn');
  if(!btn)return;
  if(CONFIG.rating_mode==='side-by-side'){
    var assignment=CONFIG.assignments[state.current_index];
    var labels=Object.keys(assignment.label_map).sort();
    btn.disabled=Object.keys(sideRankings).length!==labels.length;
  }else{
    var allRated=CONFIG.criteria.every(function(c){
      return document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
    });
    btn.disabled=!allRated;
  }
}

// ── Submit ───────────────────────────────────────────────────────────────
function submitAssessment(){
  var btn=document.getElementById('submit-btn');
  var idx=state.current_index;
  var completed=state.assessments.length;
  var isUpdate=idx<completed;
  btn.disabled=true;btn.textContent='Saving\u2026';

  var assignment=CONFIG.assignments[idx];
  var paper=CONFIG.papers[assignment.paper_index];
  var comment=(document.getElementById('comment')||{}).value||'';
  if(comment.length>2000)comment=comment.substring(0,2000);
  var now=nowISO();

  var entry;
  if(CONFIG.rating_mode==='side-by-side'){
    entry={
      paper_id:paper.id,
      rankings:Object.assign({},sideRankings),
      label_map:assignment.label_map,
      comment:comment,
      submitted_at:now
    };
  }else{
    var ratings={};
    for(var i=0;i<CONFIG.criteria.length;i++){
      var c=CONFIG.criteria[i];
      var sel=document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
      if(sel)ratings[c.key]=parseInt(sel.dataset.val);
    }
    entry={
      paper_id:paper.id,
      model:assignment.model,
      ratings:ratings,
      comment:comment,
      submitted_at:now
    };
  }

  if(isUpdate){
    var original=state.assessments[idx];
    entry.originally_submitted_at=original.originally_submitted_at||original.submitted_at;
    state.assessments[idx]=entry;
    saveState();
    btn.textContent='Updated \u2713';
    setTimeout(function(){
      btn.textContent=CONFIG.rating_mode==='side-by-side'?'Update Rankings':'Update';
      btn.disabled=false;
    },1200);
    // Re-render nav grid
    var completed2=state.assessments.length;
    var total=CONFIG.assignments.length;
    var grid=document.querySelector('.nav-grid');
    if(grid)grid.outerHTML=buildNavGrid(idx,completed2,total);
  }else{
    state.assessments.push(entry);
    state.current_index=state.assessments.length;
    saveState();
    render();
  }
}

// ── Done view ────────────────────────────────────────────────────────────
function renderDone(){
  var total=CONFIG.assignments.length;
  var completed=state.assessments.length;
  var pct=total>0?Math.round(completed/total*100):0;
  var navHtml=buildNavGrid(-1,completed,total);
  var toolbarHtml=
    '<div class="toolbar"><span>Reviewer: <strong class="name-display">'+
    escH(state.reviewer_name)+'</strong></span><div class="btn-group">'+
    '<button class="tool-btn primary" onclick="exportResults()" title="Download results JSON">'+
    '\u2913 Export JSON</button></div></div>';

  document.getElementById('app').innerHTML=
    '<div class="container"><header><h1>Text Summarization Evaluation</h1>'+toolbarHtml+
    '<div class="progress-wrap"><div class="progress-bar" style="width:'+pct+'%"></div>'+
    '<div class="progress-text">'+completed+' / '+total+' completed ('+pct+'%)</div></div>'+
    navHtml+
    '<div class="nav-legend">'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-done"></span>Completed</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-current"></span>Current</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-pending"></span>Pending</span>'+
    '</div></header><main>'+
    '<div class="card done-card"><h2>All done!</h2>'+
    '<p>You have completed all '+total+' assessments.</p>'+
    '<p style="margin-top:8px;color:var(--muted);font-size:0.85rem">'+
    'Click any item in the grid above to review or change your answers.</p>'+
    '<p style="margin-top:16px"><button onclick="exportResults()" '+
    'style="padding:12px 32px">Export Results JSON</button></p>'+
    '<p style="margin-top:12px;color:var(--text)">Thank you for your time, '+
    escH(state.reviewer_name)+'.</p></div></main></div>';
}

// ── Export ────────────────────────────────────────────────────────────────
function exportResults(){
  var result={
    name:state.reviewer_name,
    created_at:state.started_at,
    rating_mode:CONFIG.rating_mode,
    data_hash:CONFIG.data_hash,
    assignments:CONFIG.assignments,
    assessments:state.assessments.slice()
  };
  var blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url;
  a.download='evaluation_'+state.reviewer_name.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';
  document.body.appendChild(a);a.click();document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Import ───────────────────────────────────────────────────────────────
function importFromFile(file){
  if(!file)return;
  var errEl=document.getElementById('import-error');
  var reader=new FileReader();
  reader.onload=function(e){
    try{
      var data=JSON.parse(e.target.result);
      if(!data.assessments||!Array.isArray(data.assessments)){
        if(errEl){errEl.textContent='Invalid file: missing assessments.';errEl.style.display='block';}
        return;
      }
      if(data.data_hash&&data.data_hash!==CONFIG.data_hash){
        if(errEl){errEl.textContent='Warning: this export was generated from different evaluation data. Importing anyway.';
          errEl.style.display='block';}
      }
      state.reviewer_name=data.name||state.reviewer_name||'Imported';
      state.started_at=data.created_at||state.started_at||nowISO();
      state.assessments=data.assessments;
      state.current_index=data.assessments.length;
      saveState();
      render();
    }catch(ex){
      if(errEl){errEl.textContent='Failed to parse JSON file.';errEl.style.display='block';}
    }
  };
  reader.readAsText(file);
}

// ── Init ─────────────────────────────────────────────────────────────────
render();
</script>
</body>
</html>"""


# ── HTML generation ───────────────────────────────────────────────────────


def generate_html(config: dict) -> str:
    """Produce self-contained HTML with embedded evaluation data."""
    data_json = (
        json.dumps(config, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


# ── CLI ───────────────────────────────────────────────────────────────────


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
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output HTML file path",
    )
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--simple-ratings",
        action="store_true",
        default=False,
        help="use single acceptability score instead of 4 SummEval dimensions",
    )
    mode_group.add_argument(
        "--side-by-side",
        action="store_true",
        default=False,
        help="rank all 4 summaries per paper instead of rating individually",
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
        "--num-papers",
        type=int,
        default=DEFAULT_NUM_PAPERS,
        help=f"papers to select (default: {DEFAULT_NUM_PAPERS})",
    )
    p.add_argument(
        "--seed",
        type=str,
        default=None,
        help="random seed for assignment order (default: random)",
    )
    return p.parse_args()


def main() -> None:
    args = cli()

    if args.side_by_side:
        rating_mode = "side-by-side"
        criteria: list[dict] = []
    elif args.simple_ratings:
        rating_mode = "simple"
        criteria = SIMPLE_CRITERIA
    else:
        rating_mode = "detailed"
        criteria = DETAILED_CRITERIA

    eval_data = load_evaluation_data(
        args.results_file, args.goldstandard, args.models, args.num_papers
    )

    assignments = generate_assignments(eval_data, rating_mode, args.seed)

    # Strip fields not needed in the HTML (abstract, journal) to reduce size
    papers_slim = [
        {
            "id": p["id"],
            "title": p["title"],
            "reference_highlights": p["reference_highlights"],
            "summaries": p["summaries"],
        }
        for p in eval_data["papers"]
    ]

    data_hash = hashlib.sha256(
        json.dumps(papers_slim, sort_keys=True).encode()
    ).hexdigest()[:16]

    config = {
        "data_hash": data_hash,
        "rating_mode": rating_mode,
        "criteria": criteria,
        "models": eval_data["models"],
        "papers": papers_slim,
        "assignments": assignments,
    }

    html = generate_html(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    log.info("=" * 56)
    log.info("Offline Evaluation File Generated")
    log.info("-" * 56)
    log.info("Output:       %s", args.output)
    log.info("Mode:         %s", rating_mode)
    log.info("Papers:       %d", len(eval_data["papers"]))
    log.info("Models:       %s", ", ".join(args.models))
    log.info("Assessments:  %d per reviewer", len(assignments))
    log.info("Data hash:    %s", data_hash)
    if args.seed:
        log.info("Seed:         %s", args.seed)
    log.info("-" * 56)
    log.info(
        "Send this HTML file to reviewers. They open it in any "
        "browser, evaluate, and export a JSON file to send back."
    )
    log.info("=" * 56)


if __name__ == "__main__":
    main()
