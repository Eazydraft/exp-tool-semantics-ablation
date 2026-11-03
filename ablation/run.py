"""Runner: register model x condition combos in BFCL, generate, evaluate, index.

    python -m ablation.run [--models ...] [--conditions ...] [--threads N] [--jobs J] [--limit K]

Everything for one run lives under runs/<run-name>/:
    result/<registry>/...        BFCL's own result files (needed by its evaluator)
    score/<registry>/...         BFCL's own score files + leaderboard CSVs
    trajectories/<registry>/<task_id>.json   one file per trajectory (raw, complete)
    logs/<registry>.log          generation log per model x condition
    index.json                   one row per trajectory file + per-combo summary
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from ablation.conditions import CONDITIONS, registry_name
from ablation.models import BASE_MODELS, MODELS, api_model

# NOTE: bfcl_eval (and ablation.handler, which imports it) is only imported inside functions,
# after BFCL_PROJECT_ROOT is set in main(): BFCL resolves its result/score dirs at import time.

PROJECT_DIR = Path(__file__).resolve().parents[1]
BFCL_DIR = PROJECT_DIR / "gorilla" / "berkeley-function-call-leaderboard"
TEST_CATEGORY = "multi_turn_base"  # BFCL v3 base multi-turn, 200 tasks



def register_models():
    """Add every model x condition combo to BFCL's model registry (in memory only)."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    from ablation.handler import AblationHandler

    for model_id, info in MODELS.items():
        for condition_id in CONDITIONS:
            MODEL_CONFIG_MAPPING[registry_name(model_id, condition_id)] = ModelConfig(
                model_name=api_model(model_id),
                display_name=f"{model_id} [{condition_id}]",
                url=f"https://openrouter.ai/{model_id}",
                org=info["org"],
                license="see provider",
                model_handler=AblationHandler,
                input_price=info["input_price"],
                output_price=info["output_price"],
                is_fc_model=True,
                underscore_to_dot=False,
            )


def task_ids(args) -> list[str] | None:
    """Subset of task ids to run, or None for all 200."""
    if args.task_ids:
        return [f"{TEST_CATEGORY}_{i}" for i in args.task_ids]
    if args.limit:
        return [f"{TEST_CATEGORY}_{i}" for i in range(args.limit)]
    return None


# ---- generate ------------------------------------------------------------------

def generate_one(args, registry: str):
    """Run BFCL generation for one model x condition (this is what each subprocess does)."""
    register_models()
    from bfcl_eval._llm_response_generation import main as generation_main

    from ablation.handler import AblationHandler

    run_dir = Path(os.environ["BFCL_PROJECT_ROOT"])
    AblationHandler.trajectory_dir = run_dir / "trajectories"
    AblationHandler.run_info = json.loads((run_dir / "run_config.json").read_text())

    # Same namespace `bfcl generate` builds from its CLI flags; GPU/vLLM options are unused here.
    generation_main(SimpleNamespace(
        model=[registry], test_category=[TEST_CATEGORY], temperature=args.temperature,
        include_input_log=False, exclude_state_log=False, num_threads=args.threads,
        result_dir=None, allow_overwrite=False, run_ids=bool(task_ids(args)),
        num_gpus=1, gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=False,
        local_model_path=None, enable_lora=False, max_lora_rank=None, lora_modules=None,
    ))


def generate_all(args, run_dir: Path, registries: list[str]):
    """Run each model x condition in its own subprocess, `--jobs` at a time."""
    (run_dir / "logs").mkdir(exist_ok=True)
    ids = task_ids(args)
    if ids:  # written once here, read by every job (BFCL's --run-ids mechanism)
        (run_dir / "test_case_ids_to_generate.json").write_text(json.dumps({TEST_CATEGORY: ids}))

    def run(registry):
        log = run_dir / "logs" / f"{registry.replace('/', '_')}.log"
        print(f"[start] {registry}  (tail -f {log})", flush=True)
        cmd = [sys.executable, "-m", "ablation.run", "generate-one", "--registry", registry,
               "--run-name", args.run_name, "--threads", str(args.threads),
               "--temperature", str(args.temperature)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.task_ids:
            cmd += ["--task-ids", *map(str, args.task_ids)]
        with open(log, "a") as f:
            code = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok' if code == 0 else 'FAILED exit ' + str(code)}] {registry}  (log: {log})", flush=True)
        return code

    # BFCL's own progress bars live in the log files; print a summary here every 30s.
    def progress():
        while not done.wait(30):
            counts = [f"{r.split('/')[1]}={len(list((run_dir / 'trajectories' / r.replace('/', '_')).glob('*.json')))}"
                      for r in registries]
            print(f"[progress {time.strftime('%H:%M:%S')}] trajectories written: " + "  ".join(counts), flush=True)

    done = threading.Event()
    threading.Thread(target=progress, daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run, registries))
    done.set()
    if any(codes):
        raise SystemExit("some generation jobs failed, see logs")


# ---- evaluate --------------------------------------------------------------------

def evaluate(run_dir: Path, registries: list[str]):
    """Run BFCL's own evaluator, then copy each task's verdict into its trajectory file."""
    register_models()
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.utils import load_file

    # partial_eval: score whatever subset of tasks exists in the result files.
    evaluation_main(model=registries, test_categories=[TEST_CATEGORY],
                    result_dir=None, score_dir=None, partial_eval=True)

    for registry in registries:
        reg_dir = registry.replace("/", "_")
        result_file = run_dir / "result" / reg_dir / "multi_turn" / f"BFCL_v4_{TEST_CATEGORY}_result.json"
        score_file = run_dir / "score" / reg_dir / "multi_turn" / f"BFCL_v4_{TEST_CATEGORY}_score.json"
        if not (result_file.exists() and score_file.exists()):
            continue
        # Score file: first line is the accuracy header, then one entry per FAILED task.
        failures = {e["id"]: e for e in load_file(score_file)[1:]}
        for entry in load_file(result_file):
            path = run_dir / "trajectories" / reg_dir / f"{entry['id']}.json"
            if not path.exists():
                continue
            doc = json.loads(path.read_text())
            fail = failures.get(entry["id"])
            doc["evaluation"] = {"valid": fail is None, "error": fail["error"] if fail else None}
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))


# ---- index -----------------------------------------------------------------------

def build_index(run_dir: Path):
    rows = []
    for path in sorted((run_dir / "trajectories").rglob("*.json")):
        doc = json.loads(path.read_text())
        turns = doc["bfcl_result"] if isinstance(doc["bfcl_result"], list) else []
        tool_results = [m for turn in doc["bfcl_metadata"].get("inference_log", []) if isinstance(turn, dict)
                        for step in turn.values() if isinstance(step, list) for m in step if m.get("role") == "tool"]
        ev = doc.get("evaluation") or {}
        rows.append({
            "file": str(path.relative_to(run_dir)),
            "id": doc["id"], "model": doc["model"], "condition": doc["condition"],
            "valid": ev.get("valid"),
            "error_type": (ev.get("error") or {}).get("error_type"),
            "inference_error": doc["error"] is not None,
            "num_turns": len(turns),
            "num_steps": sum(len(t) for t in turns),
            "num_tool_calls": sum(len(step) for t in turns for step in t if isinstance(step, list)),
            "num_failed_tool_calls": sum(1 for r in tool_results if str(r["content"]).startswith("Error during execution")),
            **doc["totals"],
            "wall_time_s": doc["wall_time_s"],
            "started_at": doc["started_at"], "finished_at": doc["finished_at"],
        })

    summary = {}
    for r in rows:
        s = summary.setdefault(f"{r['model']} {r['condition']}", {"tasks": 0, "evaluated": 0, "passed": 0, "cost_usd": 0.0})
        s["tasks"] += 1
        s["evaluated"] += r["valid"] is not None
        s["passed"] += r["valid"] is True
        s["cost_usd"] += r["cost_usd"]
    for s in summary.values():
        s["success_rate"] = s["passed"] / s["evaluated"] if s["evaluated"] else None

    (run_dir / "index.json").write_text(json.dumps(
        {"run": json.loads((run_dir / "run_config.json").read_text()), "summary": summary, "trajectories": rows},
        indent=1))
    print(f"index: {run_dir / 'index.json'}  ({len(rows)} trajectories)")
    for name, s in summary.items():
        print(f"  {name:45s} {s['passed']}/{s['evaluated']} passed  ${s['cost_usd']:.4f}")


# ---- main ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="all", choices=["all", "generate", "evaluate", "index", "generate-one"])
    p.add_argument("--models", nargs="+", default=BASE_MODELS, choices=list(MODELS),
                   help="base models by default; the *-think/*-nothink reasoning variants on request")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--threads", type=int, default=8, help="parallel tasks per model x condition")
    p.add_argument("--jobs", type=int, default=4, help="model x condition combos run in parallel")
    p.add_argument("--limit", type=int, help="only run the first K tasks (smoke test)")
    p.add_argument("--task-ids", type=int, nargs="+", help="only run these task indices")
    p.add_argument("--temperature", type=float, default=0.001, help="BFCL default")
    p.add_argument("--run-name", default="main", help="results go to runs/<run-name>/")
    p.add_argument("--registry", help="(generate-one only) registry name, e.g. openai/gpt-6-luna-D1")
    args = p.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing from .env"

    # BFCL stores result/score/lock files under this root; must be set before bfcl_eval is imported.
    run_dir = PROJECT_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(run_dir)

    if args.command == "generate-one":
        return generate_one(args, args.registry)

    bfcl_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BFCL_DIR, capture_output=True, text=True).stdout.strip()
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": args.run_name, "models": args.models, "conditions": args.conditions,
        "temperature": args.temperature, "threads": args.threads, "limit": args.limit, "task_ids": args.task_ids,
        "test_category": TEST_CATEGORY, "bfcl_commit": bfcl_commit, "conditions_spec": CONDITIONS,
    }, indent=1))

    registries = [registry_name(m, c) for m in args.models for c in args.conditions]
    if args.command in ("all", "generate"):
        generate_all(args, run_dir, registries)
    if args.command in ("all", "evaluate"):
        evaluate(run_dir, registries)
    if args.command in ("all", "evaluate", "index"):
        build_index(run_dir)


if __name__ == "__main__":
    main()
