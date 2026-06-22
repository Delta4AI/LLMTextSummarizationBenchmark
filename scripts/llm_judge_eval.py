#!/usr/bin/env python3
"""
LLM-as-a-judge over the human-rated items only (4 models x rated papers).

For each (paper_id, model) the experts assessed, ask an LLM to rate the same
four dimensions (coherence, fluency, relevance, consistency) on the same 1-5
rubric, using the same evidence the humans saw (title+abstract as source,
gold-standard highlights as reference). Scores are cached and resumable, then
correlated against the experts via human_vs_automatic_correlation.py.

Scope (Option A): the 20 expert-rated papers x ALL models (~1,260 items). This
extends the human-eval setting to every system on the exact documents experts
saw — the validated judge stands in for expanding manual evaluation. The 80
human-rated (paper, model) items remain a reportable validation slice (the
correlation script only correlates where expert ratings exist).

Output: Output/scripts/llm_judge_scores.json
    { "<paper_id>|||<model>": {"coherence": int, ... , "_judge_model": str} }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from llm_apis.anthropic_client import AnthropicSummaryClient  # noqa: E402

DIMENSIONS = ["coherence", "fluency", "relevance", "consistency"]
JUDGE_MODEL = "claude-sonnet-4-6"  # bump to claude-opus-4-8 for the strongest judge
JUDGE_PARAMS = {"temperature": 0, "max_tokens": 300}

GH_HASH = "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
BASE_DIR = SCRIPT_DIR.parent / "Output" / "llm_summarization_benchmark" / GH_HASH
EVAL_DIR = SCRIPT_DIR.parent / "Output" / "scripts" / "human_evaluation_data"
PER_PAPER_JSON = BASE_DIR / "detailed_scores_per_paper.json"
OUT_JSON = SCRIPT_DIR.parent / "Output" / "scripts" / "llm_judge_scores.json"
EVAL_FILES = [f"evaluation_{i}.json" for i in range(1, 9)]

# rubric verbatim from human_evaluation_server.py (the definitions experts saw)
SYSTEM_PROMPT = (
    "You are an expert evaluator of scientific-paper summaries. Rate the summary "
    "on each dimension using a 1-5 integer scale (1 = very poor, 5 = excellent). "
    "Judge each dimension independently.\n\n"
    "Coherence: The summary should be well-structured and well-organized. It should "
    "not just be a heap of related information, but should build from sentence to "
    "sentence to a coherent body of information about a topic.\n"
    "Fluency: The quality of individual sentences. They should have no formatting "
    "problems, capitalization errors or obviously ungrammatical sentences (e.g., "
    "fragments, missing components) that make the text difficult to read.\n"
    "Relevance: Does the summary capture the important information present in the "
    "reference highlights? Penalize summaries which contain redundancies, miss key "
    "points from the highlights, or include excess information.\n"
    "Consistency: The factual alignment between the summary and the title and "
    "abstract. A factually consistent summary contains only statements that are "
    "entailed by the source. Penalize summaries that contain hallucinated facts not "
    "supported by the title or abstract.\n\n"
    'Respond with ONLY a JSON object: '
    '{"coherence": int, "fluency": int, "relevance": int, "consistency": int}'
)


def build_query(paper: dict) -> str:
    highlights = "\n".join(f"- {h}" for h in paper.get("summaries", []))
    return (
        f"TITLE:\n{paper['title']}\n\n"
        f"ABSTRACT:\n{paper['abstract']}\n\n"
        f"REFERENCE HIGHLIGHTS:\n{highlights}\n\n"
        f"SUMMARY TO RATE:\n{paper['extracted_response']}"
    )


def parse_ratings(text: str) -> dict[str, int]:
    """Extract and validate the 4-dimension JSON. Raises on malformed output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
    obj = json.loads(match.group(0))
    out = {}
    for dim in DIMENSIONS:
        if dim not in obj:
            raise ValueError(f"missing dimension {dim!r} in {obj}")
        val = int(obj[dim])
        if not 1 <= val <= 5:
            raise ValueError(f"{dim} out of range: {val}")
        out[dim] = val
    return out


def rated_papers(eval_dir: Path) -> set[str]:
    """Unique paper_ids the experts assessed (the 20 documents)."""
    missing = [fp for fp in EVAL_FILES if not (eval_dir / fp).exists()]
    if missing:
        sys.exit(f"Missing eval files in {eval_dir}: {missing}")
    papers: set[str] = set()
    for fp in EVAL_FILES:
        with open(eval_dir / fp) as f:
            data = json.load(f)
        for a in data["assessments"]:
            papers.add(a["paper_id"])
    return papers


def main():
    if not PER_PAPER_JSON.exists():
        sys.exit(f"Missing {PER_PAPER_JSON} — copy it from the prod run dir.")

    # Option A: the 20 expert-rated papers x ALL models (judge as scalable
    # stand-in for expanding the human-eval setting to every system).
    papers = rated_papers(EVAL_DIR)
    with open(PER_PAPER_JSON) as f:
        per_paper = json.load(f)
    paper_lookup = {
        (p["id"], model): p for model, papers_ in per_paper.items() for p in papers_
    }
    keys = {key for key in paper_lookup if key[0] in papers}

    cache: dict[str, dict] = {}
    if OUT_JSON.exists():
        cache = json.loads(OUT_JSON.read_text())

    client = AnthropicSummaryClient()
    todo = sorted(k for k in keys if f"{k[0]}|||{k[1]}" not in cache)
    print(f"{len(papers)} rated papers x {len(per_paper)} models = {len(keys)} "
          f"items; {len(cache)} cached, {len(todo)} to judge with {JUDGE_MODEL}.")

    errors = 0
    for i, key in enumerate(todo, 1):
        paper_id, model = key
        paper = paper_lookup.get(key)
        if not paper or not paper.get("extracted_response"):
            print(f"  [{i}/{len(todo)}] SKIP {model} — no summary text for {paper_id}")
            errors += 1
            continue
        try:
            text, _, _ = client.summarize(
                text=build_query(paper),
                model_name=JUDGE_MODEL,
                system_prompt_override=SYSTEM_PROMPT,
                parameter_overrides=JUDGE_PARAMS,
            )
            ratings = parse_ratings(text)
        except Exception as e:  # noqa: BLE001 — log, keep going, persist progress
            print(f"  [{i}/{len(todo)}] ERROR {model} {paper_id}: {e}")
            errors += 1
            continue

        cache[f"{paper_id}|||{model}"] = {**ratings, "_judge_model": JUDGE_MODEL}
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        print(f"  [{i}/{len(todo)}] {model}: {ratings}")

    print(f"\nDone. {len(cache)} scored, {errors} errors. Wrote {OUT_JSON}")


def demo():
    """ponytail: self-check — parser accepts valid JSON, rejects bad."""
    assert parse_ratings('{"coherence":3,"fluency":4,"relevance":2,"consistency":5}') == \
        {"coherence": 3, "fluency": 4, "relevance": 2, "consistency": 5}
    assert parse_ratings('noise {"coherence":3,"fluency":4,"relevance":2,'
                         '"consistency":5} trailing')["consistency"] == 5
    for bad in ['{"coherence":3}', '{"coherence":9,"fluency":4,"relevance":2,'
                '"consistency":5}', 'not json']:
        try:
            parse_ratings(bad)
            assert False, f"should have rejected: {bad}"
        except (ValueError, json.JSONDecodeError):
            pass
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
