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

Ratings are obtained via forced tool-use (structured output): the model must
return a schema-valid object, eliminating parse failures. A leading rationale
field lets it reason before scoring (reason-then-score preserves quality;
constraining the integers alone would not improve it).

Output: Output/scripts/llm_judge_scores.json
    { "<paper_id>|||<model>": {"coherence": int, ..., "rationale": str,
                               "_judge_model": str} }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from llm_apis.anthropic_client import AnthropicSummaryClient  # noqa: E402

DIMENSIONS = ["coherence", "fluency", "relevance", "consistency"]
JUDGE_MODEL = "claude-sonnet-4-6"  # bump to claude-opus-4-8 for the strongest judge
MAX_TOKENS = 1500  # headroom: verbose rationales truncate the tool call otherwise

# forced tool-use schema -> structured, always-valid output (no regex parsing).
# rationale first so the model reasons before committing the integers.
RATING_TOOL = {
    "name": "rate_summary",
    "description": "Record the 1-5 ratings for the summary on each dimension.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {"type": "string",
                          "description": "Justification in 1-2 sentences total "
                                         "(not per dimension)."},
            **{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMENSIONS},
        },
        "required": ["rationale", *DIMENSIONS],
    },
}

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
    "Give a brief rationale (1-2 sentences total, not per dimension), then "
    "record your four ratings using the rate_summary tool."
)


def build_query(paper: dict) -> str:
    highlights = "\n".join(f"- {h}" for h in paper.get("summaries", []))
    return (
        f"TITLE:\n{paper['title']}\n\n"
        f"ABSTRACT:\n{paper['abstract']}\n\n"
        f"REFERENCE HIGHLIGHTS:\n{highlights}\n\n"
        f"SUMMARY TO RATE:\n{paper['extracted_response']}"
    )


def validate_ratings(obj: dict) -> dict:
    """Validate the tool-call payload (schema guarantees keys/types; we still
    range-check, since Anthropic does not enforce min/max). Raises on bad input."""
    out: dict = {}
    for dim in DIMENSIONS:
        if dim not in obj:
            raise ValueError(f"missing dimension {dim!r} in {obj}")
        val = int(obj[dim])
        if not 1 <= val <= 5:
            raise ValueError(f"{dim} out of range: {val}")
        out[dim] = val
    out["rationale"] = str(obj.get("rationale", ""))
    return out


def judge_one(client, paper: dict) -> dict:
    """Force the rate_summary tool and return the validated ratings."""
    resp = client.client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        tools=[RATING_TOOL],
        tool_choice={"type": "tool", "name": "rate_summary"},
        messages=[{"role": "user", "content": build_query(paper)}],
    )
    if resp.stop_reason == "max_tokens":
        raise ValueError(f"response truncated at max_tokens={MAX_TOKENS}")
    for block in resp.content:
        if block.type == "tool_use" and block.name == "rate_summary":
            return validate_ratings(block.input)
    raise ValueError(f"no rate_summary tool call in response: {resp.stop_reason}")


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
            ratings = judge_one(client, paper)
        except Exception as e:  # noqa: BLE001 — log, keep going, persist progress
            print(f"  [{i}/{len(todo)}] ERROR {model} {paper_id}: {e}")
            errors += 1
            continue

        cache[f"{paper_id}|||{model}"] = {**ratings, "_judge_model": JUDGE_MODEL}
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        scores = {d: ratings[d] for d in DIMENSIONS}
        print(f"  [{i}/{len(todo)}] {model}: {scores}")

    print(f"\nDone. {len(cache)} scored, {errors} errors. Wrote {OUT_JSON}")


def demo():
    """ponytail: self-check — validator accepts good payloads, rejects bad."""
    good = {"coherence": 3, "fluency": 4, "relevance": 2, "consistency": 5,
            "rationale": "ok"}
    out = validate_ratings(good)
    assert {d: out[d] for d in DIMENSIONS} == {"coherence": 3, "fluency": 4,
                                               "relevance": 2, "consistency": 5}
    assert out["rationale"] == "ok"
    for bad in [{"coherence": 3},  # missing dims
                {"coherence": 9, "fluency": 4, "relevance": 2, "consistency": 5},
                {}]:
        try:
            validate_ratings(bad)
            assert False, f"should have rejected: {bad}"
        except (ValueError, KeyError):
            pass
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
