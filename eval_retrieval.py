"""Measure retrieval against a fixed question set. No model calls, no cost.

Retrieval is the ceiling on this system: the model can only be as right as the
passages it is handed, and a passage that never surfaces cannot be cited however
good the prompt is. This harness asks, for each question, whether the passage
that answers it came back at all — and whether the questions the corpus does not
cover are correctly refused.

It deliberately stops there. It does not ask the chat model anything, so it
costs ~56 Cohere calls (one embed and one rerank per case) from the trial key's
1,000 a month, runs in about a minute, and reports Hit@k and MRR you can compare
before and after a change to the floor, the chunking or the reranker.

    python eval_retrieval.py [--k 6] [--verbose] [--strict] [--no-rerank]

Cases live in evals/retrieval.json, each naming a phrase the corpus really
contains — checked against data/all_chunks.json when the case was written, not
guessed from what retrieval happened to return.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import observability
import rag
from vectorstore import display_text, token_length

CASES = Path(__file__).parent / "evals" / "retrieval.json"


def run_case(case, k, use_rerank):
    """(passed, detail) for one case."""
    question = case["question"]
    scope = case.get("scope") or rag.detect_framework(question)
    hits, best = rag.retrieve(question, k=k, framework=scope, use_rerank=use_rerank)

    if case.get("expect") == "refusal":
        # Nothing should clear the floor. `best` says how close the nearest
        # candidate came, which is what tells you the floor is set sanely rather
        # than merely rejecting everything.
        return not hits, {"scope": scope, "best": best, "hits": len(hits), "rank": None,
                          "match_rank": None, "tokens": 0}

    # Two questions, one retrieval. `rank` asks whether the answer is in what
    # the model reads — the expanded block; `match_rank` whether the search
    # itself landed on the clause that says it. Without expansion they agree.
    detail = {
        "scope": scope, "best": best, "hits": len(hits),
        "rank": _rank(case, hits, display_text),
        "match_rank": _rank(case, hits, lambda d: d.metadata.get("matched_text") or display_text(d)),
        "tokens": sum(token_length(display_text(d)) for d, _ in hits),
    }
    return detail["rank"] is not None, detail


def _rank(case, hits, text_of):
    """1-based rank of the first hit carrying the expected text, or None."""
    # Whitespace-normalised on both sides: where a PDF wraps a line is not part
    # of what a passage says, and a rebuilt corpus wraps differently.
    wanted_text = " ".join((case.get("expect_text") or "").lower().split())
    wanted_framework = case.get("expect_framework")
    for rank, (doc, _) in enumerate(hits, start=1):
        if wanted_framework and doc.metadata.get("framework_id") != wanted_framework:
            continue
        if wanted_text and wanted_text not in " ".join(text_of(doc).lower().split()):
            continue
        return rank
    return None


def ranking_metrics(ranks, k):
    """Hit@1/3/k and MRR over every answerable case, a miss counting as 0.

    `ranks` holds one entry per case: the 1-based rank of the answering passage,
    or None when it never came back. Unlike mean rank, neither number can be
    improved by a hard case failing.
    """
    if not ranks:
        return "no answerable cases"
    n = len(ranks)
    cutoffs = sorted({1, 3, k})
    hits = "  ".join(
        f"Hit@{c} {sum(1 for r in ranks if r and r <= c) / n:.0%}" for c in cutoffs
    )
    mrr = sum(1 / r for r in ranks if r) / n
    return f"{hits}  MRR {mrr:.3f}  (over all {n} answerable cases, a miss scores 0)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=rag.RETRIEVAL_K)
    parser.add_argument("--verbose", action="store_true", help="show every case, not just failures")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    parser.add_argument(
        "--no-rerank", action="store_true", help="plain vector order, for an A/B against the reranker"
    )
    args = parser.parse_args()
    use_rerank = rag.rerank_enabled() and not args.no_rerank

    cases = json.loads(CASES.read_text())["cases"]
    lookups = [c for c in cases if c.get("expect") != "refusal"]
    refusals = [c for c in cases if c.get("expect") == "refusal"]

    rerank = rag.RERANK_MODEL if use_rerank else "off"
    print(
        f"{len(cases)} cases  (k={args.k}, floor={rag.MAX_DISTANCE}, "
        f"fetch={rag.FETCH_K}, rerank={rerank})\n"
    )

    failures = []
    found_at = []
    # One entry per answerable case, None for a miss — the honest denominator
    # for Hit@k and MRR. found_at alone excludes misses and so flatters itself.
    lookup_ranks, match_ranks, tokens, blocks = [], [], [], []
    for case in cases:
        passed, detail = run_case(case, args.k, use_rerank)
        if not passed:
            failures.append((case, detail))
        elif detail["rank"]:
            found_at.append(detail["rank"])
        if case.get("expect") != "refusal":
            lookup_ranks.append(detail["rank"])
            match_ranks.append(detail["match_rank"])
            tokens.append(detail["tokens"])
            blocks.append(detail["hits"])

        if args.verbose or not passed:
            mark = "pass" if passed else "FAIL"
            where = f"rank {detail['rank']}" if detail["rank"] else f"{detail['hits']} hits"
            best = f"{detail['best']:.3f}" if detail["best"] is not None else "—"
            scope = detail["scope"] or "all"
            print(f"  {mark}  {where:<9} best={best}  scope={scope:<32} {case['question'][:58]}")

    lookup_failures = [c for c, _ in failures if c.get("expect") != "refusal"]
    refusal_failures = [c for c, _ in failures if c.get("expect") == "refusal"]

    hit_rate = (len(lookups) - len(lookup_failures)) / len(lookups) if lookups else 0
    refusal_rate = (len(refusals) - len(refusal_failures)) / len(refusals) if refusals else 0
    mean_rank = sum(found_at) / len(found_at) if found_at else 0

    print(f"\nanswerable   {len(lookups) - len(lookup_failures)}/{len(lookups)}  ({hit_rate:.0%} of questions retrieved the passage that answers them)")
    print(f"refused      {len(refusals) - len(refusal_failures)}/{len(refusals)}  ({refusal_rate:.0%} of out-of-corpus questions correctly returned nothing)")
    print(f"mean rank    {mean_rank:.1f}  (where the answering passage landed, 1 is best)")

    print(f"\nmodel reads   {ranking_metrics(lookup_ranks, args.k)}")
    if rag.EXPAND:
        print(f"search match  {ranking_metrics(match_ranks, args.k)}")
    if tokens:
        print(f"evidence      {sum(blocks) / len(blocks):.1f} sources, "
              f"{sum(tokens) / len(tokens):,.0f} tokens per answerable question (max {max(tokens):,})")

    # os._exit skips interpreter teardown, which once aborted on the way out
    # (chromadb racing torch) and turned every run into a failure for a caller
    # reading the exit code. Kept so the exit code only ever reflects the cases;
    # buffered traces are sent first, since atexit hooks will not run.
    observability.flush()
    sys.stdout.flush()
    os._exit(1 if failures and args.strict else 0)


if __name__ == "__main__":
    main()
