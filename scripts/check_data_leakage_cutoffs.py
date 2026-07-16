#!/usr/bin/env python3
"""
Data-leakage check (LOWEST EFFORT):
Compare model training-cutoff dates with benchmark article publication dates
to identify potential overlap (articles published before a model's cutoff
could theoretically be in its training data).

Outputs:
  - Output/scripts/data_leakage_cutoff_check.json   (machine-readable)
  - Output/scripts/data_leakage_cutoff_check.html   (interactive dashboard)
  - prints a human-readable summary to stdout
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CUTOFFS_PATH = ROOT / "Resources" / "model_training_cutoffs.json"
DATASET_PATH = ROOT / "Resources" / "text_summarization_goldstandard_data.json"
OUTPUT_JSON  = ROOT / "Output" / "scripts" / "data_leakage_cutoff_check.json"
OUTPUT_HTML  = ROOT / "Output" / "scripts" / "data_leakage_cutoff_check.html"

CLOSED_PLATFORMS = {"openai", "anthropic", "mistral"}

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_partial_date(d: str | None) -> datetime | None:
    """Parse YYYY, YYYY-MM, or YYYY-MM-DD strings into datetime objects."""
    if d is None:
        return None
    d = str(d).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None

def fmt_date(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m") if dt else "unknown"

# ── load data ────────────────────────────────────────────────────────────────

with open(CUTOFFS_PATH) as f:
    models_raw = json.load(f)

with open(DATASET_PATH) as f:
    articles = json.load(f)

# ── parse article publication dates ──────────────────────────────────────────

pub_dates: list[datetime] = []
articles_with_dates = 0
articles_without_dates = 0
for a in articles:
    dt = parse_partial_date(a.get("publication_date"))
    if dt:
        pub_dates.append(dt)
        articles_with_dates += 1
    else:
        articles_without_dates += 1

pub_dates.sort()
earliest_pub = pub_dates[0]
latest_pub = pub_dates[-1]

# distribution by year-month
pub_dist: dict[str, int] = defaultdict(int)
for dt in pub_dates:
    pub_dist[dt.strftime("%Y-%m")] += 1

# ── analyse each model ───────────────────────────────────────────────────────

results = []
for m in models_raw:
    cutoff_dt = parse_partial_date(m.get("training_cutoff"))
    model_name = m["model"]
    platform = m["platform"]

    if cutoff_dt is None:
        n_before = None
        n_after = None
        pct_before = None
        risk = "unknown (no cutoff date)"
    else:
        n_before = sum(1 for d in pub_dates if d <= cutoff_dt)
        n_after  = sum(1 for d in pub_dates if d >  cutoff_dt)
        pct_before = round(100 * n_before / len(pub_dates), 1)

        if pct_before == 0:
            risk = "none"
        elif pct_before <= 5:
            risk = "minimal"
        elif pct_before <= 25:
            risk = "low"
        elif pct_before <= 50:
            risk = "moderate"
        elif pct_before <= 75:
            risk = "high"
        else:
            risk = "very high"

    results.append({
        "platform": platform,
        "model": model_name,
        "training_cutoff": m.get("training_cutoff"),
        "cutoff_source": m.get("source"),
        "articles_before_cutoff": n_before,
        "articles_after_cutoff": n_after,
        "pct_potentially_leaked": pct_before,
        "risk_level": risk,
    })

# sort: highest risk first
risk_order = {"very high": 0, "high": 1, "moderate": 2, "low": 3, "minimal": 4, "none": 5, "unknown (no cutoff date)": 6}
results.sort(key=lambda r: (risk_order.get(r["risk_level"], 99), -(r["pct_potentially_leaked"] or 0)))

# ── build output dict ────────────────────────────────────────────────────────

output = {
    "description": (
        "Data-leakage plausibility check: articles published on or before "
        "a model's training-cutoff could theoretically appear in its training data."
    ),
    "benchmark_dataset": {
        "total_articles": len(articles),
        "articles_with_publication_date": articles_with_dates,
        "articles_without_publication_date": articles_without_dates,
        "earliest_publication": fmt_date(earliest_pub),
        "latest_publication": fmt_date(latest_pub),
        "publication_distribution": dict(sorted(pub_dist.items())),
    },
    "models": results,
}

# ── write JSON ───────────────────────────────────────────────────────────────

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

# ── write HTML ───────────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data_json = json.dumps(data)

    # The HTML template uses JavaScript template literals (backtick + ${}).
    # We keep it as a plain Python string and only do a single .replace()
    # on a unique placeholder to inject the JSON blob.
    html = _HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__TIMESTAMP__", timestamp)
    return html

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Leakage – Cutoff Check</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --orange: #db6d28; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.5;
  }
  h1 { font-size: 1.6rem; margin-bottom: .3rem; }
  h2 { font-size: 1.15rem; margin: 1.8rem 0 .8rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  .subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: .95rem; }

  /* ── Notice banner ── */
  .verdict {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem 1.3rem; margin-bottom: 1.5rem; font-size: .92rem; line-height: 1.6;
  }
  .verdict-warn { border-left: 4px solid var(--yellow); }
  .verdict strong { color: var(--yellow); }

  /* ── Publication chart ── */
  .chart-wrap {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem 1.3rem; margin-bottom: 1.5rem; overflow-x: auto;
  }
  .chart-title { font-size: .85rem; color: var(--muted); margin-bottom: .5rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  .bar-chart { display: flex; align-items: flex-end; gap: 3px; height: 120px; }
  .bar-col { display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 22px; }
  .bar {
    width: 100%; border-radius: 3px 3px 0 0; min-height: 2px;
    transition: opacity .15s;
  }
  .bar:hover { opacity: .8; }
  .bar-label { font-size: .55rem; color: var(--muted); margin-top: 3px; writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; max-height: 50px; }
  .bar-count { font-size: .6rem; color: var(--muted); margin-bottom: 2px; }

  /* ── Controls ── */
  .controls {
    display: flex; gap: .8rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center;
  }
  .controls input, .controls select {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: .45rem .7rem; font-size: .9rem; outline: none;
  }
  .controls input:focus, .controls select:focus { border-color: var(--accent); }
  .controls input { width: 260px; }

  /* ── Table ── */
  table {
    width: 100%; border-collapse: collapse; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    font-size: .88rem;
  }
  th {
    background: #1c2128; text-align: left; padding: .55rem .7rem;
    border-bottom: 1px solid var(--border); color: var(--muted);
    font-weight: 600; font-size: .8rem; text-transform: uppercase;
    letter-spacing: .04em; cursor: pointer; user-select: none;
    white-space: nowrap;
  }
  th:hover { color: var(--text); }
  th .arrow { margin-left: .3rem; font-size: .7rem; }
  td {
    padding: .45rem .7rem; border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }

  /* ── Badges ── */
  .badge {
    display: inline-block; padding: .1rem .45rem; border-radius: 10px;
    font-size: .7rem; font-weight: 600; vertical-align: middle;
  }
  .platform-tag {
    display: inline-block; padding: .1rem .45rem; border-radius: 4px;
    font-size: .75rem; font-weight: 600; background: #1c2128;
    border: 1px solid var(--border);
  }
  .closed-tag {
    display: inline-block; padding: .05rem .35rem; border-radius: 4px;
    font-size: .6rem; font-weight: 700; background: #2a1215; color: var(--red);
    margin-left: .3rem; vertical-align: middle; letter-spacing: .03em;
  }
  .src-tag {
    display: inline-block; padding: .05rem .4rem; border-radius: 8px;
    font-size: .65rem; font-weight: 600; letter-spacing: .02em;
    vertical-align: middle; margin-left: .35rem;
  }
  .src-tag-confirmed  { background: #0d2818; color: var(--green); }
  .src-tag-community  { background: #0d1f2d; color: var(--accent); }
  .src-tag-estimated  { background: #1a1800; color: var(--yellow); }
  .src-tag-unknown    { background: #2a1215; color: var(--red); }

  /* ── Risk pills ── */
  .risk {
    display: inline-block; padding: .15rem .55rem; border-radius: 10px;
    font-size: .72rem; font-weight: 700; white-space: nowrap;
  }
  .risk-none      { background: #0d2818; color: var(--green); }
  .risk-minimal   { background: #0d2818; color: var(--green); }
  .risk-low       { background: #1a1800; color: var(--yellow); }
  .risk-moderate  { background: #291800; color: var(--orange); }
  .risk-high      { background: #2a1215; color: var(--red); }
  .risk-very-high { background: #3d0d10; color: #ff7b72; }
  .risk-unknown   { background: #1c2128; color: var(--muted); }

  /* ── Overlap bar ── */
  .overlap-bar-wrap {
    display: flex; align-items: center; gap: .5rem; min-width: 160px;
  }
  .overlap-bar-track {
    flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;
    min-width: 80px;
  }
  .overlap-bar-fill {
    height: 100%; border-radius: 4px; transition: width .3s;
  }
  .overlap-pct { font-size: .8rem; font-variant-numeric: tabular-nums; min-width: 42px; text-align: right; }

  .num-cell { font-variant-numeric: tabular-nums; text-align: right; }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    body { padding: 1rem; }
  }
</style>
</head>
<body>

<h1>Data Leakage – Training Cutoff Check</h1>
<p class="subtitle">Generated by <code>scripts/check_data_leakage_cutoffs.py</code> · __TIMESTAMP__</p>

<div id="unknownBox"></div>

<h2>Benchmark Article Publication Distribution</h2>
<div class="chart-wrap">
  <div class="chart-title">Articles per month (n = <span id="totalArticles"></span>)</div>
  <div class="bar-chart" id="barChart"></div>
</div>

<h2>Per-Model Overlap Analysis</h2>
<p class="subtitle" style="margin-bottom:.8rem">Articles published <em>on or before</em> a model's training cutoff could theoretically appear in its training data.
Closed-source models are flagged with <span class="closed-tag">CLOSED</span> since their training corpora are not auditable.</p>

<div class="controls">
  <input type="text" id="search" placeholder="Filter models…">
  <select id="platformFilter">
    <option value="">All platforms</option>
  </select>
  <select id="riskFilter">
    <option value="">All risk levels</option>
    <option value="very high">🔴 Very high</option>
    <option value="high">🔴 High</option>
    <option value="moderate">🟠 Moderate</option>
    <option value="low">🟡 Low</option>
    <option value="minimal">🟢 Minimal</option>
    <option value="none">🟢 None</option>
    <option value="unknown">❓ Unknown</option>
  </select>
  <select id="sourceFilter">
    <option value="">All sources</option>
    <option value="closed">Closed-source only</option>
    <option value="open">Open-source only</option>
  </select>
</div>

<table>
<thead>
<tr>
  <th data-col="0">Platform <span class="arrow"></span></th>
  <th data-col="1">Model <span class="arrow"></span></th>
  <th data-col="2">Training Cutoff <span class="arrow"></span></th>
  <th data-col="3">≤ Cutoff <span class="arrow"></span></th>
  <th data-col="4">> Cutoff <span class="arrow"></span></th>
  <th data-col="5">Potential Overlap <span class="arrow"></span></th>
  <th data-col="6">Risk <span class="arrow"></span></th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>

<script>
// ── Data (embedded by Python) ──
const DATA = __DATA_PLACEHOLDER__;

const CLOSED_PLATFORMS = new Set(["openai", "anthropic", "mistral"]);
const RISK_ORDER = {"very high":0,"high":1,"moderate":2,"low":3,"minimal":4,"none":5,"unknown (no cutoff date)":6};

const ds = DATA.benchmark_dataset;
const models = DATA.models;
const unknownModels = models.filter(m => m.pct_potentially_leaked === null);

document.getElementById("totalArticles").textContent = ds.total_articles;

// ── Unknown-cutoff notice ──
if (unknownModels.length) {
  document.getElementById("unknownBox").innerHTML =
    `<div class="verdict verdict-warn">⚠️ <strong>${unknownModels.length} model(s) have no known training-cutoff date</strong> — data leakage cannot be assessed for these models: ` +
    unknownModels.map(m => m.model).join(", ") + `</div>`;
}

// ── Bar chart ──
const dist = ds.publication_distribution;
const months = Object.keys(dist).sort();
const maxCount = Math.max(...Object.values(dist));
const chartEl = document.getElementById("barChart");

// Compute the latest closed-source cutoff dynamically
const closedCutoffs = models
  .filter(m => CLOSED_PLATFORMS.has(m.platform) && m.training_cutoff)
  .map(m => m.training_cutoff)
  .sort();
const latestClosedCutoff = closedCutoffs.length ? closedCutoffs[closedCutoffs.length - 1] : "0000";

months.forEach(m => {
  const count = dist[m];
  const h = Math.max(2, (count / maxCount) * 110);
  const color = m <= latestClosedCutoff ? "var(--yellow)" : "var(--green)";
  const col = document.createElement("div");
  col.className = "bar-col";
  col.innerHTML = `<span class="bar-count">${count}</span><div class="bar" style="height:${h}px;background:${color}" title="${m}: ${count} articles"></div><span class="bar-label">${m}</span>`;
  chartEl.appendChild(col);
});

// ── Table rows ──
const tbody = document.getElementById("tbody");
const platforms = [...new Set(models.map(m => m.platform))].sort();
const platformSel = document.getElementById("platformFilter");
platforms.forEach(p => {
  const opt = document.createElement("option");
  opt.value = p; opt.textContent = p;
  platformSel.appendChild(opt);
});

function riskClass(level) {
  const l = level.replace(/\s*\(.*/, "").replace(/\s+/g, "-");
  return `risk-${l}`;
}
function riskLabel(level) {
  return level.replace(" (no cutoff date)", "");
}
function overlapColor(pct) {
  if (pct === null) return "var(--muted)";
  if (pct <= 5) return "var(--green)";
  if (pct <= 15) return "var(--yellow)";
  if (pct <= 30) return "var(--orange)";
  return "var(--red)";
}
function srcTagClass(src) {
  if (src === "confirmed") return "src-tag-confirmed";
  if (src === "community_dataset") return "src-tag-community";
  if (src === "estimated") return "src-tag-estimated";
  return "src-tag-unknown";
}
function srcLabel(src) {
  if (src === "community_dataset") return "community";
  return src || "unknown";
}

models.forEach(m => {
  const isClosed = CLOSED_PLATFORMS.has(m.platform);
  const closedBadge = isClosed ? '<span class="closed-tag">CLOSED</span>' : '';
  const pct = m.pct_potentially_leaked;
  const before = m.articles_before_cutoff;
  const after = m.articles_after_cutoff;
  const pctStr = pct !== null ? pct + "%" : "?";
  const barWidth = pct !== null ? pct : 0;

  const tr = document.createElement("tr");
  tr.dataset.platform = m.platform;
  tr.dataset.risk = m.risk_level;
  tr.dataset.closed = isClosed ? "1" : "0";

  tr.innerHTML = `
    <td data-sort="${m.platform}"><span class="platform-tag">${m.platform}</span></td>
    <td data-sort="${m.model}">${m.model}${closedBadge}</td>
    <td data-sort="${m.training_cutoff || '9999'}" class="date-cell" style="white-space:nowrap">
      ${m.training_cutoff
        ? `<span style="font-variant-numeric:tabular-nums;font-weight:600">${m.training_cutoff}</span>`
        : `<span style="color:var(--muted);font-style:italic">unknown</span>`}
      <span class="src-tag ${srcTagClass(m.cutoff_source)}">${srcLabel(m.cutoff_source)}</span>
    </td>
    <td data-sort="${before !== null ? String(before).padStart(5,'0') : '99999'}" class="num-cell">${before !== null ? before : '?'}</td>
    <td data-sort="${after !== null ? String(after).padStart(5,'0') : '99999'}" class="num-cell">${after !== null ? after : '?'}</td>
    <td data-sort="${pct !== null ? String(pct*10).padStart(5,'0') : '99999'}">
      <div class="overlap-bar-wrap">
        <span class="overlap-pct" style="color:${overlapColor(pct)}">${pctStr}</span>
        <div class="overlap-bar-track">
          <div class="overlap-bar-fill" style="width:${barWidth}%;background:${overlapColor(pct)}"></div>
        </div>
      </div>
    </td>
    <td data-sort="${RISK_ORDER[m.risk_level] !== undefined ? RISK_ORDER[m.risk_level] : 9}">
      <span class="risk ${riskClass(m.risk_level)}">${riskLabel(m.risk_level)}</span>
    </td>
  `;
  tbody.appendChild(tr);
});

// ── Filtering ──
const allRows = [...tbody.querySelectorAll("tr")];
const searchEl = document.getElementById("search");
const riskFilterEl = document.getElementById("riskFilter");
const sourceFilterEl = document.getElementById("sourceFilter");

function applyFilters() {
  const q = searchEl.value.toLowerCase();
  const plat = platformSel.value;
  const risk = riskFilterEl.value;
  const src = sourceFilterEl.value;
  allRows.forEach(r => {
    const text = r.textContent.toLowerCase();
    const show = text.includes(q)
      && (!plat || r.dataset.platform === plat)
      && (!risk || r.dataset.risk.startsWith(risk))
      && (!src || (src === "closed" ? r.dataset.closed === "1" : r.dataset.closed === "0"));
    r.style.display = show ? "" : "none";
  });
}
searchEl.addEventListener("input", applyFilters);
platformSel.addEventListener("change", applyFilters);
riskFilterEl.addEventListener("change", applyFilters);
sourceFilterEl.addEventListener("change", applyFilters);

// ── Sorting ──
let sortCol = -1, sortAsc = true;
document.querySelectorAll("th[data-col]").forEach(th => {
  th.addEventListener("click", () => {
    const col = +th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc; else { sortCol = col; sortAsc = true; }
    document.querySelectorAll("th .arrow").forEach(a => a.textContent = "");
    th.querySelector(".arrow").textContent = sortAsc ? "▲" : "▼";
    allRows.sort((a, b) => {
      const av = a.children[col].dataset.sort || a.children[col].textContent.trim();
      const bv = b.children[col].dataset.sort || b.children[col].textContent.trim();
      return sortAsc ? av.localeCompare(bv, undefined, {numeric:true}) : bv.localeCompare(av, undefined, {numeric:true});
    });
    allRows.forEach(r => tbody.appendChild(r));
  });
});
</script>
</body>
</html>'''

with open(OUTPUT_HTML, "w") as f:
    f.write(generate_html(output))

# ── pretty-print summary to stdout ───────────────────────────────────────────

print("=" * 80)
print("DATA-LEAKAGE CUTOFF CHECK")
print("=" * 80)
print(f"\nBenchmark dataset : {len(articles)} articles")
print(f"  with pub date   : {articles_with_dates}")
print(f"  without pub date: {articles_without_dates}")
print(f"  date range      : {fmt_date(earliest_pub)} → {fmt_date(latest_pub)}")
print()

print("-" * 80)
print(f"{'Model':<52} {'Cutoff':>9}  {'≤cutoff':>7} {'%leak':>6}  Risk")
print("-" * 80)
for r in results:
    tag = " [C]" if r["platform"] in CLOSED_PLATFORMS else ""
    name = f"{r['model']}{tag}"
    if len(name) > 52:
        name = name[:49] + "..."
    cutoff = r["training_cutoff"] or "n/a"
    n_b = str(r["articles_before_cutoff"]) if r["articles_before_cutoff"] is not None else "?"
    pct = f"{r['pct_potentially_leaked']}%" if r["pct_potentially_leaked"] is not None else "?"
    print(f"{name:<52} {cutoff:>9}  {n_b:>7} {pct:>6}  {r['risk_level']}")

print("-" * 80)
print()

# Summary stats
for label, platforms in [("CLOSED-SOURCE", CLOSED_PLATFORMS), ("OPEN-SOURCE", set(r["platform"] for r in results) - CLOSED_PLATFORMS)]:
    subset = [r for r in results if r["platform"] in platforms]
    if not subset:
        continue
    known = [r for r in subset if r["pct_potentially_leaked"] is not None]
    unknown = [r for r in subset if r["pct_potentially_leaked"] is None]
    print(f"  {label} ({len(subset)} models):")
    if known:
        avg_pct = sum(r["pct_potentially_leaked"] for r in known) / len(known)
        max_r = max(known, key=lambda r: r["pct_potentially_leaked"])
        min_r = min(known, key=lambda r: r["pct_potentially_leaked"])
        print(f"    avg potential overlap : {avg_pct:.1f}%")
        print(f"    max overlap           : {max_r['pct_potentially_leaked']}% ({max_r['model']})")
        print(f"    min overlap           : {min_r['pct_potentially_leaked']}% ({min_r['model']})")
    if unknown:
        print(f"    unknown cutoff        : {len(unknown)} model(s): {', '.join(u['model'] for u in unknown)}")
    print()

# Final verdict
all_known = [r for r in results if r["pct_potentially_leaked"] is not None]
any_overlap = any(r["pct_potentially_leaked"] > 0 for r in all_known)
if any_overlap:
    print("⚠  OVERLAP EXISTS – some benchmark articles were published before")
    print("   model training cutoffs. Data leakage is theoretically possible")
    print("   (especially for closed-source models whose training data is unknown).")
    high_risk = [r for r in results if r["risk_level"] in ("high", "very high") and r["platform"] in CLOSED_PLATFORMS]
    if high_risk:
        print(f"\n   Closed-source models with HIGH/VERY HIGH risk:")
        for r in high_risk:
            print(f"     • {r['model']} (cutoff {r['training_cutoff']}, {r['pct_potentially_leaked']}% overlap)")
else:
    print("✓  NO OVERLAP – all benchmark articles were published after every")
    print("   model's training cutoff. Data leakage via pre-training is implausible.")

unknown_models = [r for r in results if r["risk_level"].startswith("unknown")]
if unknown_models:
    print(f"\n⚠  {len(unknown_models)} model(s) have NO KNOWN cutoff date – leakage cannot be ruled out:")
    for r in unknown_models:
        print(f"     • {r['model']} ({r['platform']})")

print(f"\nOutputs:")
print(f"  JSON → {OUTPUT_JSON.relative_to(ROOT)}")
print(f"  HTML → {OUTPUT_HTML.relative_to(ROOT)}")
