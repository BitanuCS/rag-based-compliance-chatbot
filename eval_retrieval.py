"""Measure retrieval against a fixed question set. No model calls, no cost.

Retrieval is the ceiling on this system: the model can only be as right as the
passages it is handed, and a passage that never surfaces cannot be cited however
good the prompt is. This harness asks, for each question, whether the passage
that answers it came back at all — and whether the questions the corpus does not
cover are correctly refused.

It deliberately stops there. It does not ask the model anything, so it costs
nothing, runs in under a minute, and reports one number you can compare before
and after a change to the floor, the chunking or the ingest repairs.

    python eval_retrieval.py [--k 6] [--verbose] [--strict]

Cases live in evals/retrieval.json, each naming a phrase the corpus really
contains — checked against data/all_chunks.json when the case was written, not
guessed from what retrieval happened to return.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import rag
from vectorstore import display_text

CASES = Path(__file__).parent / "evals" / "retrieval.json"


def run_case(case, k):
    """(passed, detail) for one case."""
    question = case["question"]
    scope = case.get("scope") or rag.detect_framework(question)
    hits, best = rag.retrieve(question, k=k, framework=scope)

    if case.get("expect") == "refusal":
        # Nothing should clear the floor. `best` says how close the nearest
        # candidate came, which is what tells you the floor is set sanely rather
        # than merely rejecting everything.
        return not hits, {"scope": scope, "best": best, "hits": len(hits), "rank": None}

    wanted_text = (case.get("expect_text") or "").lower()
    wanted_framework = case.get("expect_framework")
    for rank, (doc, _) in enumerate(hits, start=1):
        if wanted_framework and doc.metadata.get("framework_id") != wanted_framework:
            continue
        if wanted_text and wanted_text not in display_text(doc).lower():
            continue
        return True, {"scope": scope, "best": best, "hits": len(hits), "rank": rank}
    return False, {"scope": scope, "best": best, "hits": len(hits), "rank": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=rag.RETRIEVAL_K)
    parser.add_argument("--verbose", action="store_true", help="show every case, not just failures")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())["cases"]
    lookups = [c for c in cases if c.get("expect") != "refusal"]
    refusals = [c for c in cases if c.get("expect") == "refusal"]

    print(f"{len(cases)} cases  (k={args.k}, floor={rag.MAX_DISTANCE}, fetch={rag.FETCH_K})\n")

    failures = []
    found_at = []
    for case in cases:
        passed, detail = run_case(case, args.k)
        if not passed:
            failures.append((case, detail))
        elif detail["rank"]:
            found_at.append(detail["rank"])

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

    # chromadb and torch race each other through interpreter teardown and abort
    # on the way out, which would turn every run into a failure for a caller
    # reading the exit code. The work is finished and flushed by this point.
    sys.stdout.flush()
    os._exit(1 if failures and args.strict else 0)


if __name__ == "__main__":
    main()
