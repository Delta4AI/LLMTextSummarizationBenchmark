#!/usr/bin/env python3
"""
LLM-as-a-judge panel over the human-rated papers x ALL models.

For each (paper_id, model) the experts assessed, ask THREE judges from distinct
providers to rate the same four dimensions (coherence, fluency, relevance,
consistency) on the same 1-5 rubric, using the same evidence the humans saw
(title+abstract as source, gold-standard highlights as reference). Scores are
cached and resumable, then correlated against the experts via
human_vs_automatic_correlation.py.

Why a panel: a single judge from one provider invites a self-preference attack —
a judge can inflate summaries from its own model family. Three judges from
distinct providers (Anthropic, OpenAI, Mistral) let the panel median stand in as
the headline score AND make the bias measurable as a per-provider delta. All
three are mid-tier peers (Sonnet's tier), with reasoning minimized so none
ruminates its way to a different scale.

Scope (Option A): the 20 expert-rated papers x ALL models (~1,260 items) x 3
judges. This extends the human-eval setting to every system on the exact
documents experts saw — the validated panel stands in for expanding manual
evaluation. The 80 human-rated (paper, model) items remain a reportable
validation slice (the correlation script only correlates where expert ratings
exist).

Ratings are obtained via forced structured output (Anthropic forced tool-use,
OpenAI/Mistral Pydantic-schema parse): the model must return a schema-valid
object, eliminating parse failures. A leading rationale field lets it reason
before scoring (reason-then-score preserves quality).

Output: Output/scripts/llm_judge_scores.json
    { "<paper_id>|||<model>|||<judge_key>": {"coherence": int, ..., "rationale":
      str, "_judge_model": str, "_judge": str} }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from llm_apis.anthropic_client import AnthropicSummaryClient  # noqa: E402

DIMENSIONS = ["coherence", "fluency", "relevance", "consistency"]
MAX_TOKENS = 1500  # headroom: verbose rationales truncate the tool call otherwise
OPENAI_MAX_OUTPUT_TOKENS = 3000  # reasoning(low) tokens + structured output share this

# forced tool-use schema (Anthropic) -> structured, always-valid output.
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


class Ratings(BaseModel):
    """Structured judge output (OpenAI/Mistral Pydantic schema). Field order =
    rationale first, so the model reasons before committing the four integers."""
    rationale: str
    coherence: int
    fluency: int
    relevance: int
    consistency: int

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


def judge_anthropic(client, paper: dict, model: str) -> dict:
    """Force the rate_summary tool and return the validated ratings."""
    resp = client.messages.create(
        model=model,
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


def judge_openai(client, paper: dict, model: str) -> dict:
    """Pydantic-schema structured output with reasoning pinned low. GPT-5.x
    reasoning models reject temperature, so it is omitted (effort=low is the
    'don't overthink' control)."""
    resp = client.responses.parse(
        model=model,
        reasoning={"effort": "low"},
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_query(paper)},
        ],
        text_format=Ratings,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise ValueError(f"no parsed output (status={resp.status})")
    return validate_ratings(parsed.model_dump())


def judge_mistral(client, paper: dict, model: str) -> dict:
    """Pydantic-schema structured output at temperature 0. prompt_mode is left
    unset (default) so Medium 3.5 does not engage extended reasoning."""
    resp = client.chat.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_query(paper)},
        ],
        response_format=Ratings,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError("no parsed output from Mistral")
    return validate_ratings(parsed.model_dump())


def _anthropic_client():
    return AnthropicSummaryClient().client


def _openai_client():
    from llm_apis.openai_client import OpenAISummaryClient
    return OpenAISummaryClient().client


def _mistral_client():
    from llm_apis.mistral_client import MistralSummaryClient
    return MistralSummaryClient().client


# Three mid-tier peers (Sonnet's tier) from distinct providers, reasoning
# minimized. All pinned to dated snapshots so published numbers stay reproducible.
JUDGES = [
    {"key": "anthropic", "model": "claude-sonnet-4-6",
     "make": _anthropic_client, "judge": judge_anthropic},
    {"key": "openai", "model": "gpt-5.4-2026-03-05",
     "make": _openai_client, "judge": judge_openai},
    {"key": "mistral", "model": "mistral-medium-2604",
     "make": _mistral_client, "judge": judge_mistral},
]


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


def cache_key(paper_id: str, model: str, judge_key: str) -> str:
    return f"{paper_id}|||{model}|||{judge_key}"


def run_judge(spec: dict, todo: list[tuple[str, str]], paper_lookup: dict,
              cache: dict) -> tuple[int, int]:
    """Score every pending item for one judge, persisting after each. Returns
    (scored, errors). A failed client init skips the whole judge."""
    jkey, model = spec["key"], spec["model"]
    try:
        client = spec["make"]()
    except Exception as e:  # noqa: BLE001 — missing key / SDK error -> skip judge
        print(f"  SKIP judge {jkey} ({model}): client init failed: {e}")
        return 0, len(todo)

    scored = errors = 0
    for i, key in enumerate(todo, 1):
        paper_id, model_sys = key
        paper = paper_lookup.get(key)
        if not paper or not paper.get("extracted_response"):
            print(f"  [{jkey} {i}/{len(todo)}] SKIP {model_sys} — no summary "
                  f"for {paper_id}")
            errors += 1
            continue
        try:
            ratings = spec["judge"](client, paper, model)
        except Exception as e:  # noqa: BLE001 — log, keep going, persist progress
            print(f"  [{jkey} {i}/{len(todo)}] ERROR {model_sys} {paper_id}: {e}")
            errors += 1
            continue

        cache[cache_key(paper_id, model_sys, jkey)] = {
            **ratings, "_judge_model": model, "_judge": jkey}
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        scored += 1
        scores = {d: ratings[d] for d in DIMENSIONS}
        print(f"  [{jkey} {i}/{len(todo)}] {model_sys}: {scores}")
    return scored, errors


def main():
    if not PER_PAPER_JSON.exists():
        sys.exit(f"Missing {PER_PAPER_JSON} — copy it from the prod run dir.")

    # Option A: the 20 expert-rated papers x ALL models (panel as scalable
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

    print(f"{len(papers)} rated papers x {len(per_paper)} models = {len(keys)} "
          f"items x {len(JUDGES)} judges. {len(cache)} cached.")

    total_scored = total_errors = 0
    for spec in JUDGES:
        todo = sorted(k for k in keys
                      if cache_key(k[0], k[1], spec["key"]) not in cache)
        if not todo:
            print(f"\n{spec['key']} ({spec['model']}): all cached, skipping.")
            continue
        print(f"\n{spec['key']} ({spec['model']}): {len(todo)} to judge.")
        scored, errors = run_judge(spec, todo, paper_lookup, cache)
        total_scored += scored
        total_errors += errors

    print(f"\nDone. {len(cache)} total ratings cached "
          f"({total_scored} new, {total_errors} errors). Wrote {OUT_JSON}")


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

    # Pydantic structured-output path (OpenAI/Mistral) round-trips to a valid payload
    r = Ratings(rationale="ok", coherence=3, fluency=4, relevance=2, consistency=5)
    assert validate_ratings(r.model_dump()) == out
    assert list(Ratings.model_fields) == ["rationale", *DIMENSIONS]  # rationale first

    # judge-scoped cache keys are distinct per provider
    assert cache_key("p1", "gpt-4o", "openai") != cache_key("p1", "gpt-4o", "mistral")
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
