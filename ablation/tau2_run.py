"""The same documentation conditions on tau2-bench's airline domain (run through ./run.sh tau2).

    python -m ablation.tau2_run [--models ...] [--conditions ...] [--trials 2 3 4] [--jobs J] [--threads N] [--limit K]

tau2-bench is used unmodified from the `tau2-bench/` submodule (text mode, airline domain, 50 tasks,
its default step limits; one trial by default, further trials with --trials, each with its own seed). The agent is tau2's own LLMAgent with two changes: the
tool schemas pass through `apply_condition` on the way out and the model's calls are translated back
on the way in, so the environment and the evaluator see original names. The user is simulated by
gpt-4.1 as in the tau2 paper, and the communicate-info judge is tau2 2.0's pinned gpt-5.5; both are
reached through OpenRouter like the agent. Every LLM call is recorded with OpenRouter's usage.

Plays are written next to the BFCL trajectories of the same run, in the same layout:
    runs/<run-name>/trajectories/<registry>/<task_id>.json   tau2's SimulationRun (messages, reward),
                                 the tools sent, the name maps and every raw LLM call
    runs/<run-name>/index.json   shared with BFCL (ablation.index)
"""

import argparse
import datetime
import json
import os
import shutil
import threading
import time
from multiprocessing import get_context
from pathlib import Path

from dotenv import load_dotenv

from ablation.conditions import apply_condition, restore_call
from ablation.index import build_index
from ablation.models import BASE_MODELS, MODELS, api_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOMAIN = "airline"
USER_MODEL = "openrouter/openai/gpt-4.1"            # tau2 paper's user simulator (gpt-4.1-2025-04-14)
COMMUNICATE_JUDGE = "openrouter/openai/gpt-5.5"     # tau2 2.0's pinned semantic communicate-info judge
NL_JUDGE = "openrouter/openai/gpt-5.4-mini"         # tau2 2.0's NL-assertion judge (not in airline's reward basis)
CONDITIONS = ["D0", "D1", "D2", "D3", "D4", "D5"]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---- tau2 patches: record every LLM call, take cost from OpenRouter, route judges via OpenRouter ----

_tls = threading.local()


def install_patches():
    """Idempotent: a pool worker runs several combos."""
    import tau2.evaluator.evaluator_nl_assertions as nl
    import tau2.runner.batch as batch
    import tau2.utils.llm_utils as lu

    if getattr(lu, "_ablation_patched", False):
        return
    lu._ablation_patched = True
    original_completion = lu.completion

    def recording_completion(**kwargs):
        t0 = time.time()
        response = original_completion(**kwargs)
        calls = getattr(_tls, "calls", None)
        if calls is not None:
            calls.append({"sent_at": now_iso(), "latency_s": time.time() - t0,
                          "request": {k: v for k, v in kwargs.items() if k != "num_retries"},
                          "response": response.to_dict()})
        return response

    lu.completion = recording_completion
    # litellm has no price table for OpenRouter models; OpenRouter reports the cost itself.
    lu.get_response_cost = lambda response: (response.to_dict().get("usage") or {}).get("cost") or 0.0
    nl.DEFAULT_LLM_NL_ASSERTIONS = NL_JUDGE

    original_run_single_task = batch.run_single_task

    def run_single_task(config, task, **kwargs):
        _tls.calls = []  # one simulation runs start to finish on one thread, judges included
        sim = original_run_single_task(config, task, **kwargs)
        sim.info = {**(sim.info or {}), "api_calls": _tls.calls}
        _tls.calls = None
        return sim

    batch.run_single_task = run_single_task


# ---- the agent --------------------------------------------------------------------------

class AblatedTool:
    """What tau2's generate() needs from a tool: a name and an OpenAI schema."""

    def __init__(self, function: dict):
        self.name = function["name"]
        self.openai_schema = {"type": "function", "function": function}


def make_agent_class():
    from tau2.agent.llm_agent import LLMAgent

    class AblationAgent(LLMAgent):
        """tau2's LLMAgent shown ablated schemas; its calls are translated back for the environment."""

        def __init__(self, tools, domain_policy, llm, llm_args, condition):
            super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
            functions, self.maps = apply_condition([t.openai_schema["function"] for t in tools], condition)
            self.tools = [AblatedTool(f) for f in functions]

        def generate_next_message(self, message, state):
            assistant = self._generate_next_message(message, state)
            state.messages.append(assistant)  # the model's own history keeps the names it was shown
            restored = assistant.model_copy(deep=True)
            for tc in restored.tool_calls or []:
                tc.name, args = restore_call(tc.name, json.dumps(tc.arguments), self.maps)
                tc.arguments = json.loads(args)
            return restored, state

    return AblationAgent


def register_agent():
    from tau2.registry import registry

    if registry.get_agent_factory("ablation_agent") is not None:
        return
    AblationAgent = make_agent_class()

    def factory(tools, domain_policy, llm, llm_args, **_):
        llm_args = dict(llm_args or {})
        return AblationAgent(tools, domain_policy, llm, llm_args, llm_args.pop("ablation_condition"))

    registry.register_agent_factory(factory, "ablation_agent")


# ---- one model x condition -------------------------------------------------------------------

def registry_dir(model_id: str, condition: str) -> str:
    return f"{model_id.replace('/', '_')}-{condition}"


def tools_sent(condition: str) -> tuple[list, dict]:
    from tau2.registry import registry
    env = registry.get_env_constructor(DOMAIN)()
    functions, maps = apply_condition([t.openai_schema["function"] for t in env.get_tools()], condition)
    return [{"type": "function", "function": f} for f in functions], maps


def totals(calls: list, model: str | None = None) -> dict:
    usages = [c["response"].get("usage") or {} for c in calls if model is None or c["request"]["model"] == model]
    return {"api_calls": len(usages),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
            "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for u in usages),
            "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for u in usages),
            "cost_usd": sum(u.get("cost", 0) or 0 for u in usages)}


def run_combo(job: tuple) -> dict:
    """Run every task of one model x condition through tau2's batch runner, then fold each
    SimulationRun into one trajectory file (runs in a worker process)."""
    model_id, condition, trial, threads, limit, task_ids, run_dir = job
    from tau2.data_model.semantic_communication import CommunicateScoring, SemanticCommunicateSettings
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import _load_run_tasks, run_tasks

    install_patches()
    register_agent()
    agent_model = f"openrouter/{api_model(model_id)}"
    config = TextRunConfig(
        domain=DOMAIN, agent="ablation_agent", llm_agent=agent_model,
        llm_args_agent={"temperature": 0.0, "ablation_condition": condition,
                        "extra_body": {"usage": {"include": True}, **MODELS[model_id].get("extra_body", {})}},
        user="user_simulator", llm_user=USER_MODEL,
        llm_args_user={"temperature": 0.0, "extra_body": {"usage": {"include": True}}},
        communicate_scoring=CommunicateScoring(judge=SemanticCommunicateSettings(
            model=COMMUNICATE_JUDGE, model_args={"reasoning_effort": "low"})),
        num_trials=1, max_concurrency=threads, log_level="ERROR", task_ids=task_ids, auto_resume=True,
        save_to=registry_dir(model_id, condition),
        # Trial 1 ran with tau2's own random seed (recorded per file); later trials get a fixed, distinct seed.
        seed=None if trial == 1 else 1000 * trial,
    )
    suffix = "" if trial == 1 else f".t{trial}"  # trial 1 keeps <task_id>.json, trial n writes <task_id>.t<n>.json
    tasks = _load_run_tasks(config)[:limit]
    out_dir = run_dir / "trajectories" / registry_dir(model_id, condition)
    tasks = [t for t in tasks if not (out_dir / f"{t.id}{suffix}.json").exists()]
    if not tasks:
        return {"model": model_id, "condition": condition, "trial": trial, "tasks": 0, "reward": None}
    tmp_dir = run_dir / ".tau2_tmp" / f"{registry_dir(model_id, condition)}{suffix}"  # tau2's own checkpoint (resumable)
    results = run_tasks(config, tasks, save_path=tmp_dir / "results.json", save_dir=tmp_dir, console_display=False)

    sent, maps = tools_sent(condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = []
    for sim in results.simulations:
        calls = (sim.info or {}).pop("api_calls", [])
        doc = {
            "id": sim.task_id, "benchmark": "tau2", "domain": DOMAIN, "model": model_id, "condition": condition, "trial": trial, "seed": sim.seed,
            "user_model": USER_MODEL, "communicate_judge": COMMUNICATE_JUDGE, "extra_body": MODELS[model_id].get("extra_body", {}),
            "started_at": sim.start_time, "finished_at": sim.end_time, "wall_time_s": sim.duration,
            "totals": totals(calls, agent_model), "user_totals": totals(calls, USER_MODEL),
            "judge_totals": totals([c for c in calls if c["request"]["model"] not in (agent_model, USER_MODEL)]),
            "result": {"reward": sim.reward_info.reward if sim.reward_info else None,
                       "termination_reason": str(sim.termination_reason),
                       "reward_info": sim.reward_info.model_dump(mode="json") if sim.reward_info else None},
            "tools_sent": sent, "name_maps": maps,
            "tau2": sim.model_dump(mode="json"),
            "api_calls": calls,
        }
        (out_dir / f"{sim.task_id}{suffix}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
        rewards.append(doc["result"]["reward"] or 0)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"model": model_id, "condition": condition, "trial": trial, "tasks": len(rewards), "reward": sum(rewards) / len(rewards)}


# ---- main --------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "index"])
    p.add_argument("--models", nargs="+", default=BASE_MODELS, choices=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--trials", nargs="+", type=int, default=[1], help="trial numbers to run, e.g. 2 3 4 for three more trials")
    p.add_argument("--jobs", type=int, default=4, help="model x condition combos run in parallel (processes)")
    p.add_argument("--threads", type=int, default=8, help="simulations in flight per combo")
    p.add_argument("--limit", type=int, help="only the first K tasks (smoke test)")
    p.add_argument("--task-ids", nargs="+", help="only these task ids")
    p.add_argument("--run-name", default="main", help="results go to runs/<run-name>/trajectories/")
    args = p.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing from .env"
    run_dir = PROJECT_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        jobs = [(m, c, t, args.threads, args.limit, args.task_ids, run_dir) for m in args.models for c in args.conditions for t in args.trials]
        print(f"{len(jobs)} model x condition combos (tasks with a trajectory file are skipped)", flush=True)
        with get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            for r in pool.imap_unordered(run_combo, jobs):
                reward = "nothing to do" if r["reward"] is None else f"mean reward {r['reward']:.3f} over {r['tasks']} tasks"
                print(f"[{time.strftime('%H:%M:%S')}] {r['model']} {r['condition']} trial {r['trial']}: {reward}", flush=True)
        shutil.rmtree(run_dir / ".tau2_tmp", ignore_errors=True)
    build_index(run_dir)


if __name__ == "__main__":
    main()
