"""Run the evaluation cases against the real AI model and write evals/REPORT.md.

Each case's result is saved as soon as it finishes, so a run that stops (for example when the
free AI allowance runs out for the day) carries on where it left off next time.

    uv run python scripts/run_eval.py                      # current prompts, label "current"
    uv run python scripts/run_eval.py --prompts baseline   # the first prompts, label "baseline"
    uv run python scripts/run_eval.py --only P01 Q03       # just these cases
    uv run python scripts/run_eval.py --report             # rewrite the report from saved results
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from procurement_agent import prompts
from procurement_agent.agent import ProcurementAgent
from procurement_agent.company_data import load_company_data
from procurement_agent.evaluation import Score, score, summary
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import GroqClient
from procurement_agent.tools import ToolBox

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
RESULTS = EVALS / "results"
BUSY = "busy or unavailable"
LABELS = {"baseline": "First prompts", "current": "Current prompts"}


def load_cases() -> list[dict]:
    return json.loads((EVALS / "cases.json").read_text())["cases"]


def use_baseline_prompts() -> None:
    spec = importlib.util.spec_from_file_location("baseline", EVALS / "baseline_prompts.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    for name in ("UNDERSTAND", "INVESTIGATE", "DRAFT", "ANSWER"):
        setattr(prompts, name, getattr(baseline, name))


def run(label: str, only: list[str], rerun: bool) -> int:
    folder = RESULTS / label
    folder.mkdir(parents=True, exist_ok=True)
    agent = ProcurementAgent(GroqClient(), ToolBox(), KnowledgeBase())
    for case in load_cases():
        path = folder / f"{case['id']}.json"
        if (only and case["id"] not in only) or (path.exists() and not rerun):
            continue
        started = time.perf_counter()
        response = agent.run(case["text"]).model_dump()
        seconds = round(time.perf_counter() - started, 1)
        if response["status"] == "failed" and BUSY in response["summary"]:
            print(f"{case['id']}: the AI allowance is used up for now. Run again later to resume.")
            return 2
        path.write_text(
            json.dumps(
                {"case": case["id"], "seconds": seconds, "response": response},
                indent=1,
                default=str,
            )
        )
        print(f"{case['id']}: {response['status']} in {seconds}s, {response['tokens_used']} tokens")
    return 0


def scores_for(label: str) -> list[Score]:
    data = load_company_data()
    cases = {c["id"]: c for c in load_cases()}
    found = []
    for path in sorted((RESULTS / label).glob("*.json")):
        saved = json.loads(path.read_text())
        found.append(score(cases[saved["case"]], saved["response"], saved["seconds"], data))
    return found


def write_report() -> None:
    labels = [label for label in LABELS if (RESULTS / label).exists()]
    runs = {label: scores_for(label) for label in labels}
    total = len(load_cases())
    lines = [
        "# Evaluation report",
        "",
        (
            f"Written {datetime.now(UTC):%d %B %Y}. Real AI model (`openai/gpt-oss-120b` on Groq, "
            "with `openai/gpt-oss-20b` as backup). Cases are in `evals/cases.json`; purchase answers "
            "are worked out from the policy rules in code. Rerun with `scripts/run_eval.py`."
        ),
        "",
        "| Measure | " + " | ".join(LABELS[label] for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    summaries = {label: summary(scores) for label, scores in runs.items()}
    for measure in next(iter(summaries.values()), {}):
        lines.append(f"| {measure} | " + " | ".join(summaries[lb][measure] for lb in labels) + " |")
    for label, scores in runs.items():
        lines += ["", f"## {LABELS[label]}: cases not fully correct", ""]
        if len(scores) < total:
            lines.append(f"Only {len(scores)} of {total} cases have run so far.\n")
        misses = [s for s in scores if not s.passed]
        lines += [f"- **{s.case_id}** ({s.status}): {'; '.join(s.problems)}" for s in misses]
        if not misses:
            lines.append("None.")
    (EVALS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {EVALS.relative_to(ROOT) / 'REPORT.md'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", choices=["current", "baseline"], default="current")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--rerun", action="store_true", help="run cases that already have results")
    parser.add_argument("--report", action="store_true", help="only rewrite the report")
    args = parser.parse_args()
    code = 0
    if not args.report:
        if args.prompts == "baseline":
            use_baseline_prompts()
        code = run(args.prompts, args.only, args.rerun)
    write_report()
    return code


if __name__ == "__main__":
    sys.exit(main())
