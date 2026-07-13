from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_mcmc_cigp.agent_backend.llm_client import CompatibleResponsesClient
from pc_mcmc_cigp.agent_backend.skills import SkillRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline and optional live prompt-skill regression checks")
    parser.add_argument(
        "--live", action="store_true", help="Send cases sequentially to the configured compatible API"
    )
    parser.add_argument("--cases", default="evals/agent_prompt_cases.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    runtime = SkillRuntime()
    failures = []
    for case in cases:
        skill = runtime.route(case["stage"], case["message"])
        if skill.name != case["expected_skill"]:
            failures.append(f"{case['id']}: routed to {skill.name}")
        runtime.schema(skill)  # also verifies packaged JSON is readable
        if args.live:
            result = CompatibleResponsesClient.from_env().chat(
                [{"role": "user", "content": case["message"]}],
                {"stage": case["stage"], "workflow_mode": "coupled", "evaluation_risk": case["risk"]},
            )
            if result["active_skill"] != case["expected_skill"]:
                failures.append(f"{case['id']}: live skill mismatch")
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "skill": result["active_skill"],
                        "repaired": result["schema_repaired"],
                    },
                    ensure_ascii=False,
                )
            )
    print(json.dumps({"cases": len(cases), "failures": failures, "live": args.live}, ensure_ascii=False))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
