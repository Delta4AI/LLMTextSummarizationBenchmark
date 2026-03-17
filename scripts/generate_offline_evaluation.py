#!/usr/bin/env python3
"""Generate a self-contained offline evaluation HTML file.

Creates an HTML file that reviewers can open in any browser to evaluate
summaries using a two-step SummEval flow. Progress is saved in the
browser's localStorage and can be exported as JSON.

Usage
-----
    python generate_offline_evaluation.py \\
        openai_gpt-4o anthropic_claude-opus-4-20250514 \\
        local:textrank ollama_gemma3:270M \\
        -o evaluation_alice.html
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
    eval_data: dict, seed: str | None = None
) -> list[dict]:
    """Create a shuffled assignment list (same logic as the server)."""
    rng = random.Random(seed)
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
}
[data-theme="dark"]{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;
--yellow:#d29922;--btn-text:#0d1117}
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
textarea{width:100%;min-height:60px;padding:8px 10px;background:var(--bg);
border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:0.85rem;font-family:inherit;resize:vertical;outline:none}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
</style>
</head>
<body>
<div class="modal-overlay" id="info-modal" onclick="if(event.target===this)closeInfo()">
<div class="modal" id="info-modal-body"></div>
</div>
<button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
<div id="app"></div>
<script type="application/json" id="eval-data">__DATA_JSON__</script>
<script>
// ── Config & State ───────────────────────────────────────────────────────
var CONFIG=JSON.parse(document.getElementById('eval-data').textContent);
var STORAGE_KEY='offline_eval_'+CONFIG.data_hash;
var state=loadState();

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
function openInfo(){
  var ref='<p class="ref">Fabbri, A.&nbsp;R. et&nbsp;al. (2021). SummEval: Re-evaluating Summarization Evaluation. <em>Transactions of the Association for Computational Linguistics</em>, 9, 391&ndash;409.</p>';
  var html;
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
  var info=n+' papers \u00d7 '+m+' models = '+(n*m)+' assessments';
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

// ── Two-step state ──────────────────────────────────────────────────────
var STEP1_KEYS=CONFIG.criteria.filter(function(c){return c.key!=='consistency';});
var STEP2_KEYS=CONFIG.criteria.filter(function(c){return c.key==='consistency';});
var currentStep=1;
var pendingRatings={};
var pendingComment='';

// ── Assessment view ──────────────────────────────────────────────────────
function renderAssessment(idx){
  var completed=state.assessments.length;
  var total=CONFIG.assignments.length;
  if(idx<0)idx=0;
  if(idx>completed)idx=completed;
  if(idx>=total){renderDone();return;}
  state.current_index=idx;
  currentStep=1;pendingRatings={};pendingComment='';
  saveState();

  var previous=idx<completed?state.assessments[idx]:null;
  if(previous&&previous.ratings){
    pendingRatings={};
    for(var key in previous.ratings)pendingRatings[key]=previous.ratings[key];
    pendingComment=previous.comment||'';
  }
  renderStep1();
}

function buildHeader(){
  var idx=state.current_index;
  var completed=state.assessments.length;
  var total=CONFIG.assignments.length;
  var pct=total>0?Math.round(completed/total*100):0;
  var navHtml=buildNavGrid(idx,completed,total);
  return '<header><h1>Text Summarization Evaluation</h1>'+
    '<div class="toolbar"><span>Reviewer: <strong class="name-display">'+
    escH(state.reviewer_name)+'</strong></span><div class="btn-group">'+
    '<button class="tool-btn primary" onclick="exportResults()" title="Download results JSON">'+
    '\u2913 Export JSON</button>'+
    '<label class="tool-btn" style="cursor:pointer" title="Import previous export">'+
    '\u2912 Import<input type="file" accept=".json" onchange="importFromFile(this.files[0])" '+
    'style="display:none"></label></div></div>'+
    '<div class="progress-wrap"><div class="progress-bar" style="width:'+pct+'%"></div>'+
    '<div class="progress-text">'+completed+' / '+total+' completed ('+pct+'%)</div></div>'+
    navHtml+
    '<div class="nav-legend">'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-done"></span>Completed</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-current"></span>Current</span>'+
    '<span class="nav-legend-item"><span class="nav-swatch sw-pending"></span>Pending</span>'+
    '</div></header>';
}

function buildCriteriaHTML(criteriaList){
  var html='';
  for(var i=0;i<criteriaList.length;i++){
    var c=criteriaList[i];
    html+='<div class="criterion"><div class="criterion-header">'+
      '<span class="criterion-label">'+escH(c.label)+'</span>'+
      '<span class="criterion-desc">'+escH(c.description)+'</span></div><div class="likert">';
    for(var v=1;v<=5;v++){
      var anchor=(c.anchors&&c.anchors[String(v)])||'';
      var sel=pendingRatings[c.key]===v?' selected':'';
      html+='<label class="'+sel.trim()+'" data-key="'+c.key+'" data-val="'+v+'" onclick="selectRating(this)">'+
        v+(anchor?'<span class="anchor">'+escH(anchor)+'</span>':'')+'</label>';
    }
    html+='</div></div>';
  }
  return html;
}

function renderStep1(){
  currentStep=1;
  var idx=state.current_index;
  var total=CONFIG.assignments.length;
  var completed=state.assessments.length;
  var assignment=CONFIG.assignments[idx];
  var paper=CONFIG.papers[assignment.paper_index];
  var isSubmitted=idx<completed;
  var statusBadge=isSubmitted?'<span class="status-badge completed">\u2714 completed</span>':'';
  var stepTag='<span class="step-indicator">Step 1 of 2</span>';
  var bullets=paper.reference_highlights.map(function(h){return '<li>'+escH(h)+'</li>';}).join('');
  var summary=paper.summaries[assignment.model];
  var criteriaHTML=buildCriteriaHTML(STEP1_KEYS);

  document.getElementById('app').innerHTML=
    '<div class="container">'+buildHeader()+'<main>'+
    '<div class="card"><div class="card-header">Assessment '+(idx+1)+' of '+total+
    statusBadge+stepTag+'</div>'+
    '<div class="card-header" style="margin-top:8px">Reference Highlights</div>'+
    '<ul class="highlights">'+bullets+'</ul></div>'+
    '<div class="card"><div class="card-header">Generated Summary</div>'+
    '<div class="gen-text">'+escH(summary)+'</div></div>'+
    '<div class="card"><div class="card-header">Rate Structure &amp; Content'+
    ' <button class="info-btn" onclick="openInfo()" title="About these criteria">i</button></div>'+
    criteriaHTML+
    '<div class="submit-row"><button id="next-btn" onclick="goToStep2()" disabled>Next \u2192</button></div></div>'+
    '</main></div>';
  checkStep1();
}

function renderStep2(){
  currentStep=2;
  var idx=state.current_index;
  var total=CONFIG.assignments.length;
  var completed=state.assessments.length;
  var assignment=CONFIG.assignments[idx];
  var paper=CONFIG.papers[assignment.paper_index];
  var isSubmitted=idx<completed;
  var btnLabel=isSubmitted?'Update':'Submit';
  var statusBadge=isSubmitted?'<span class="status-badge completed">\u2714 completed</span>':'';
  var stepTag='<span class="step-indicator">Step 2 of 2</span>';
  var summary=paper.summaries[assignment.model];
  var criteriaHTML=buildCriteriaHTML(STEP2_KEYS);

  document.getElementById('app').innerHTML=
    '<div class="container">'+buildHeader()+'<main>'+
    '<div class="card"><div class="card-header">Assessment '+(idx+1)+' of '+total+
    statusBadge+stepTag+'</div>'+
    '<div style="margin-bottom:10px"><div class="card-header">Publication Title</div>'+
    '<div class="gen-text">'+escH(paper.title)+'</div></div>'+
    '<div style="margin-bottom:10px"><div class="card-header">Abstract</div>'+
    '<div class="abstract-text">'+escH(paper.abstract)+'</div></div>'+
    '<div><div class="card-header">Generated Summary</div>'+
    '<div class="gen-text">'+escH(summary)+'</div></div></div>'+
    '<div class="card"><div class="card-header">Rate Factual Consistency'+
    ' <button class="info-btn" onclick="openInfo()" title="About these criteria">i</button></div>'+
    criteriaHTML+
    '<div style="margin-top:8px"><label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:4px">'+
    'Comment (optional)</label>'+
    '<textarea id="comment" placeholder="Any observations\u2026"></textarea></div>'+
    '<div class="btn-row"><button class="btn-back" onclick="goToStep1()">\u2190 Back</button>'+
    '<button id="submit-btn" onclick="submitAssessment()" disabled>'+btnLabel+'</button></div></div>'+
    '</main></div>';
  var ta=document.getElementById('comment');
  if(ta&&pendingComment)ta.value=pendingComment;
  checkStep2();
}

function goToStep2(){
  for(var i=0;i<STEP1_KEYS.length;i++){
    var c=STEP1_KEYS[i];
    var sel=document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
    if(sel)pendingRatings[c.key]=parseInt(sel.dataset.val);
  }
  renderStep2();
}

function goToStep1(){
  var sel=document.querySelector('.likert label.selected[data-key="consistency"]');
  if(sel)pendingRatings.consistency=parseInt(sel.dataset.val);
  var ta=document.getElementById('comment');
  if(ta)pendingComment=ta.value;
  renderStep1();
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
  renderAssessment(index);
}

// ── Rating interactions ──────────────────────────────────────────────────
function selectRating(el){
  var key=el.dataset.key;
  el.parentElement.querySelectorAll('label').forEach(function(l){l.classList.remove('selected');});
  el.classList.add('selected');
  if(currentStep===1)checkStep1();else checkStep2();
}

function checkStep1(){
  var btn=document.getElementById('next-btn');
  if(!btn)return;
  btn.disabled=!STEP1_KEYS.every(function(c){
    return document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
  });
}

function checkStep2(){
  var btn=document.getElementById('submit-btn');
  if(!btn)return;
  btn.disabled=!STEP2_KEYS.every(function(c){
    return document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
  });
}

// ── Submit ───────────────────────────────────────────────────────────────
function submitAssessment(){
  var btn=document.getElementById('submit-btn');
  var idx=state.current_index;
  var completed=state.assessments.length;
  var isUpdate=idx<completed;
  btn.disabled=true;btn.textContent='Saving\u2026';

  // Collect Step 2 consistency
  for(var i=0;i<STEP2_KEYS.length;i++){
    var c=STEP2_KEYS[i];
    var sel=document.querySelector('.likert label.selected[data-key="'+c.key+'"]');
    if(sel)pendingRatings[c.key]=parseInt(sel.dataset.val);
  }

  var assignment=CONFIG.assignments[idx];
  var paper=CONFIG.papers[assignment.paper_index];
  var comment=(document.getElementById('comment')||{}).value||'';
  if(comment.length>2000)comment=comment.substring(0,2000);
  var now=nowISO();

  var entry={
    paper_id:paper.id,
    model:assignment.model,
    ratings:{},
    comment:comment,
    submitted_at:now
  };
  for(var key in pendingRatings)entry.ratings[key]=pendingRatings[key];

  if(isUpdate){
    var original=state.assessments[idx];
    entry.originally_submitted_at=original.originally_submitted_at||original.submitted_at;
    state.assessments[idx]=entry;
    saveState();
    btn.textContent='Updated \u2713';
    setTimeout(function(){
      btn.textContent='Update';
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

    rating_mode = "detailed"
    criteria = DETAILED_CRITERIA

    eval_data = load_evaluation_data(
        args.results_file, args.goldstandard, args.models, args.num_papers
    )

    assignments = generate_assignments(eval_data, args.seed)

    papers_slim = [
        {
            "id": p["id"],
            "title": p["title"],
            "abstract": p["abstract"],
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
