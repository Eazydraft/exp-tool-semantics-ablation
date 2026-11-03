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
import fcntl
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

# NOTE: bfcl_eval (and ablation.handler, which imports it) is only imported inside functions,
# after BFCL_PROJECT_ROOT is set in main(): BFCL resolves its result/score dirs at import time.

PROJECT_DIR = Path(__file__).resolve().parents[1]
BFCL_DIR = PROJECT_DIR / "gorilla" / "berkeley-function-call-leaderboard"
TEST_CATEGORY = "multi_turn_base"  # BFCL v3 base multi-turn, 200 tasks

# OpenRouter model ids. Prices (USD per 1M tokens, as listed on OpenRouter on 2026-11-02) are only
# used by BFCL's leaderboard CSV; the actual cost of every call is recorded from OpenRouter's usage field.
MODELS = {
    "openai/gpt-6-luna": dict(org="OpenAI", input_price=0.10, output_price=0.50),
    "anthropic/claude-haiku-5.5": dict(org="Anthropic", input_price=0.10, output_price=0.50),
    "qwen/qwen3-30b-a3b-instruct-2507": dict(org="Qwen", input_price=0.10, output_price=0.30),
    "google/gemma-4-26b-a4b-it": dict(org="Google", input_price=0.064, output_price=0.191),
}


def register_models():
    """Add every model x condition combo to BFCL's model registry (in memory only)."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    from ablation.handler import AblationHandler

    for model_id, info in MODELS.items():
        for condition_id in CONDITIONS:
            MODEL_CONFIG_MAPPING[registry_name(model_id, condition_id)] = ModelConfig(
                model_name=model_id,
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
    # Passed by the parent rather than read from run_config.json, which a concurrent launch into the
    # same run directory may have changed since (that mislabelled the main run's trajectories).
    if "ABLATION_LAUNCH" not in os.environ:
        raise SystemExit("generate-one is started by `generate`/`all`, which pass it the launch record; run those instead")
    AblationHandler.run_info = json.loads(os.environ["ABLATION_LAUNCH"])

    # Same namespace `bfcl generate` builds from its CLI flags; GPU/vLLM options are unused here.
    generation_main(SimpleNamespace(
        model=[registry], test_category=[TEST_CATEGORY], temperature=args.temperature,
        include_input_log=False, exclude_state_log=False, num_threads=args.threads,
        result_dir=None, allow_overwrite=False, run_ids=bool(task_ids(args)),
        num_gpus=1, gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=False,
        local_model_path=None, enable_lora=False, max_lora_rank=None, lora_modules=None,
    ))


def generate_all(args, run_dir: Path, registries: list[str], launch: dict):
    """Run each model x condition in its own subprocess, `--jobs` at a time."""
    (run_dir / "logs").mkdir(exist_ok=True)
    env = {**os.environ, "ABLATION_LAUNCH": json.dumps({**launch, "conditions_spec": CONDITIONS})}
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
            code = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
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


# ---- run config ----------------------------------------------------------------

def record_launch(args, run_dir: Path):
    """Append this launch to run_config.json; models/conditions are the union over all launches.

    A run directory is usually filled by several launches (main study, follow-ups, redos), so the
    config must accumulate rather than describe only the latest one. Returns this launch's record.
    """
    bfcl_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BFCL_DIR, capture_output=True, text=True).stdout.strip()
    launch = {
        "run_name": args.run_name, "command": args.command, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": args.models, "conditions": args.conditions,
        "temperature": args.temperature, "threads": args.threads, "limit": args.limit, "task_ids": args.task_ids,
        "test_category": TEST_CATEGORY, "bfcl_commit": bfcl_commit,
    }
    path = run_dir / "run_config.json"
    # Concurrent launches into one run directory would otherwise drop each other's records.
    with open(run_dir / ".run_config.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = json.loads(path.read_text()) if path.exists() else {}
        if "launches" not in cfg:  # pre-launch-history format: the file itself describes one launch
            cfg = {"launches": [{k: v for k, v in cfg.items() if k != "conditions_spec"}] if cfg else []}
        launches = cfg["launches"]
        old_commits = {l["bfcl_commit"] for l in launches} - {bfcl_commit}
        if old_commits:
            raise SystemExit(f"{path} has results from BFCL commit {', '.join(old_commits)}, not {bfcl_commit}; "
                             "use a new --run-name so one run never mixes BFCL versions")
        launches.append(launch)
        # update() keeps any other top-level keys, such as a hand-written provenance note.
        cfg.update({
            "run_name": args.run_name,
            "models": list(dict.fromkeys(m for l in launches for m in l["models"])),
            "conditions": list(dict.fromkeys(c for l in launches for c in l["conditions"])),
            "bfcl_commit": bfcl_commit,
            "conditions_spec": CONDITIONS,
        })
        path.write_text(json.dumps(cfg, indent=1))
    return launch


# ---- main ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="all", choices=["all", "generate", "evaluate", "index", "generate-one"])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
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

    registries = [registry_name(m, c) for m in args.models for c in args.conditions]
    if args.command in ("all", "generate"):
        generate_all(args, run_dir, registries, record_launch(args, run_dir))
    if args.command in ("all", "evaluate"):
        evaluate(run_dir, registries)
    if args.command in ("all", "evaluate", "index"):
        build_index(run_dir)


if __name__ == "__main__":
    main()
