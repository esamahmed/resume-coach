"""
eval/harness.py

Evaluation harness for the Resume Coach pipeline.
Runs fixtures through the pipeline, scores each agent with the JudgeAgent,
aggregates results, and exits with code 1 if any agent fails.

Usage:
    # Dry run — validate fixtures only, no LLM calls
    python -m eval.harness --dry-run

    # Mock mode — test harness plumbing, no LLM calls
    python -m eval.harness --mock

    # Full eval with live judge scoring
    python -m eval.harness

    # Single fixture
    python -m eval.harness --fixture RC-001-java-to-ai-engineer

Exit codes:
    0 — all agents passed
    1 — one or more agents failed or critical failure detected
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FIXTURES_DIR = Path("eval/fixtures")
SCORES_DIR   = Path("eval/scores")
PASS_THRESHOLD = 7.5


def load_fixtures(fixture_name: str | None = None) -> list[dict]:
    """Load one or all fixtures from eval/fixtures/."""
    if not FIXTURES_DIR.exists():
        logger.error("Fixtures directory not found: %s", FIXTURES_DIR)
        sys.exit(1)

    pattern = f"{fixture_name}.json" if fixture_name else "*.json"
    files = sorted(FIXTURES_DIR.glob(pattern))

    if not files:
        logger.error("No fixtures found matching: %s", pattern)
        sys.exit(1)

    fixtures = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            data["_fixture_name"] = f.stem
            fixtures.append(data)
            logger.info("Loaded fixture: %s", f.stem)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in fixture %s: %s", f.name, e)
            sys.exit(1)

    return fixtures


def validate_fixture(fixture: dict) -> list[str]:
    """Validate fixture structure. Returns list of errors."""
    errors = []
    required_keys = ["input", "expected"]
    for key in required_keys:
        if key not in fixture:
            errors.append(f"Missing required key: '{key}'")

    input_keys = ["resume_text", "jd_text"]
    for key in input_keys:
        if key not in fixture.get("input", {}):
            errors.append(f"Missing input key: '{key}'")

    return errors


def run_pipeline(fixture: dict) -> dict | None:
    """
    Run the resume coach pipeline on a fixture.
    Returns the pipeline output dict or None on failure.
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from app.graph import run_pipeline as graph_run_pipeline

        resume_text = fixture["input"]["resume_text"]
        jd_text     = fixture["input"]["jd_text"]
        session_id  = f"eval-{fixture['_fixture_name']}"

        logger.info("Running pipeline for fixture: %s", fixture["_fixture_name"])
        start = time.time()

        # run_pipeline returns (state, faiss_index) — we only need state
        state, _ = graph_run_pipeline(
            resume_text=resume_text,
            jd_text=jd_text,
            session_id=session_id,
        )

        elapsed = time.time() - start
        logger.info("Pipeline complete in %.1fs", elapsed)
        return dict(state)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        return None


def score_pipeline(
    pipeline_output: dict,
    fixture: dict,
    mock: bool = False,
) -> dict[str, dict]:
    """
    Score all agents in the pipeline output using JudgeAgent.
    Returns dict of agent_name → score dict.
    """
    from app.agents.judge_agent import JudgeAgent

    judge = JudgeAgent(mock=mock)
    resume_text = fixture["input"]["resume_text"]
    jd_text     = fixture["input"]["jd_text"]

    results = judge.evaluate_pipeline(
        pipeline_output=pipeline_output,
        input_resume=resume_text,
        input_jd=jd_text,
    )

    return {name: result.to_dict() for name, result in results.items()}


def aggregate_results(scores: dict[str, dict]) -> dict:
    """Aggregate per-agent scores into a summary."""
    if not scores:
        return {"overall_pass": False, "reason": "No scores produced"}

    all_passed       = all(s["pass"] for s in scores.values())
    any_critical     = any(s["critical_failure"] for s in scores.values())
    failed_agents    = [name for name, s in scores.items() if not s["pass"]]
    critical_agents  = [name for name, s in scores.items() if s["critical_failure"]]
    avg_score        = sum(s["weighted_score"] for s in scores.values()) / len(scores)

    return {
        "overall_pass":     all_passed and not any_critical,
        "average_score":    round(avg_score, 2),
        "failed_agents":    failed_agents,
        "critical_failures": critical_agents,
        "agent_scores":     {
            name: {
                "weighted_score": s["weighted_score"],
                "pass":           s["pass"],
                "critical_failure": s["critical_failure"],
            }
            for name, s in scores.items()
        },
    }


def save_report(report: dict, fixture_name: str) -> Path:
    """Save score report to eval/scores/."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCORES_DIR / f"{fixture_name}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info("Score report saved: %s", path)
    return path


def print_summary(fixture_name: str, summary: dict, scores: dict) -> None:
    """Print a human-readable summary to stdout."""
    status = "✅ PASS" if summary["overall_pass"] else "❌ FAIL"
    print(f"\n{'='*60}")
    print(f"Fixture: {fixture_name}")
    print(f"Overall: {status} | Avg score: {summary['average_score']}/10")
    print(f"{'='*60}")

    for agent, score in scores.items():
        agent_status = "✅" if score["pass"] else "❌"
        critical     = " ⚠️  CRITICAL FAILURE" if score["critical_failure"] else ""
        print(f"  {agent_status} {agent:<25} {score['weighted_score']:.1f}/10{critical}")

        # Print dimension scores
        for dim, val in score.get("dimensions", {}).items():
            if val is not None:
                print(f"       {dim:<20} {val['score']:.1f}  {val['rationale'][:60]}...")

    if summary["failed_agents"]:
        print(f"\nFailed agents: {', '.join(summary['failed_agents'])}")
    if summary["critical_failures"]:
        print(f"Critical failures: {', '.join(summary['critical_failures'])}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume Coach eval harness")
    parser.add_argument("--fixture",  help="Run a single fixture by name")
    parser.add_argument("--dry-run",  action="store_true", help="Validate fixtures only")
    parser.add_argument("--mock",     action="store_true", help="Use mock judge scores")
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixture)
    overall_exit_code = 0

    for fixture in fixtures:
        fixture_name = fixture["_fixture_name"]
        logger.info("=" * 50)
        logger.info("Processing fixture: %s", fixture_name)

        # ── Validate fixture structure ─────────────────────────────────────
        errors = validate_fixture(fixture)
        if errors:
            logger.error("Fixture validation failed: %s", errors)
            overall_exit_code = 1
            continue

        if args.dry_run:
            logger.info("Dry run — fixture valid: %s", fixture_name)
            continue

        # ── Run pipeline ───────────────────────────────────────────────────
        pipeline_output = run_pipeline(fixture)
        if pipeline_output is None:
            logger.error("Pipeline failed for fixture: %s", fixture_name)
            overall_exit_code = 1
            continue

        # ── Score with judge ───────────────────────────────────────────────
        scores  = score_pipeline(pipeline_output, fixture, mock=args.mock)
        summary = aggregate_results(scores)

        # ── Save and print ─────────────────────────────────────────────────
        report = {
            "fixture":   fixture_name,
            "timestamp": datetime.now().isoformat(),
            "summary":   summary,
            "scores":    scores,
        }
        save_report(report, fixture_name)
        print_summary(fixture_name, summary, scores)

        if not summary["overall_pass"]:
            overall_exit_code = 1

    # ── Final exit ─────────────────────────────────────────────────────────
    if overall_exit_code == 0:
        logger.info("✅ All fixtures passed")
    else:
        logger.error("❌ One or more fixtures failed — see reports in eval/scores/")

    return overall_exit_code


if __name__ == "__main__":
    sys.exit(main())
