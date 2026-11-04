"""The same documentation conditions on Apple's ToolSandbox (run through ./run.sh toolsandbox).

    python -m ablation.toolsandbox_run [--models ...] [--conditions ...] [--jobs N] [--limit K]

ToolSandbox is used unmodified from the `toolsandbox/` submodule. We play the 91 base scenarios
that need no RapidAPI key, each in its `_all_tools` variant (every sandbox tool is offered, minus
the five RapidAPI ones), so that tool selection is part of the task as it is in BFCL. The user is
simulated by gpt-4o-2024-05-13 as in the ToolSandbox paper, reached through OpenRouter like the
agents. The agent is the stock OpenAI-API agent with two changes: the tool schemas pass through
`apply_condition` on the way out, and the model's calls are translated back on the way in.

Plays are written next to the BFCL trajectories of the same run, in the same layout:
    runs/<run-name>/trajectories/<registry>/<scenario>.json   raw API calls, tools sent, name maps,
                                 score, plus ToolSandbox's own conversation, execution context and
                                 pretty-printed transcript under "toolsandbox"
    runs/<run-name>/index.json   shared with BFCL (ablation.index)
"""

import argparse
import datetime
import json
import re
import os
import random
import shutil
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import NOT_GIVEN, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

from ablation.conditions import apply_condition, restore_call
from ablation.index import build_index
from ablation.models import BASE_MODELS, MODELS, api_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
USER_MODEL = "openai/gpt-4o-2024-05-13"
CONDITIONS = ["D0", "D1", "D2", "D3", "D4", "D5"]  # D6/D7 are BFCL-specific follow-ups
# These tools call RapidAPI and need a paid key; scenarios that need them are skipped and the
# tools are withheld everywhere else so no condition can stumble into a key error.
RAPID_TOOLS = {"search_lat_lon", "search_location_around_lat_lon", "search_weather_around_lat_lon",
               "search_stock", "convert_currency"}


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def recording_client(calls: list) -> OpenAI:
    """OpenRouter client that appends every raw request/response pair to `calls`."""
    def on_request(request):
        headers = {k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in request.headers.items()}
        calls.append({"request": {"sent_at": now_iso(), "method": request.method, "url": str(request.url),
                                  "headers": headers, "body": _json_or_text(request.content)}})

    def on_response(response):
        response.read()
        calls[-1]["response"] = {"received_at": now_iso(), "status_code": response.status_code,
                                 "headers": dict(response.headers), "body": _json_or_text(response.content)}
        calls[-1]["latency_s"] = response.elapsed.total_seconds()

    return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL, max_retries=2,
                  timeout=httpx.Timeout(180, connect=10),
                  http_client=httpx.Client(event_hooks={"request": [on_request], "response": [on_response]}))


def _json_or_text(content: bytes):
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content.decode("utf-8", errors="replace")


def totals(calls: list) -> dict:
    usages = [c["response"]["body"].get("usage") or {} for c in calls
              if isinstance(c.get("response", {}).get("body"), dict)]
    return {
        "api_calls": len(calls),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usages),
        "cost_usd": sum(u.get("cost", 0) or 0 for u in usages),
        "latency_s": sum(c.get("latency_s", 0) for c in calls),
    }


# ---- roles -------------------------------------------------------------------------

class Agent(OpenAIAPIAgent):
    """ToolSandbox's OpenAI-API agent, pointed at OpenRouter, with the condition applied to the tools."""

    def __init__(self, model_id: str, condition: str):
        self.model_name, self.condition = api_model(model_id), condition
        self.extra_body = MODELS[model_id].get("extra_body", {})
        self.calls, self.raw_calls, self.tools_sent, self.maps = [], {}, None, {"tool": {}, "param": {}}
        self.openai_client = recording_client(self.calls)

    def model_inference(self, openai_messages, openai_tools):
        tools = NOT_GIVEN
        if openai_tools is not NOT_GIVEN:
            functions, self.maps = apply_condition([t["function"] for t in openai_tools], self.condition)
            tools = self.tools_sent = [{"type": "function", "function": f} for f in functions]
        with all_logging_disabled():
            response = self.openai_client.chat.completions.create(
                model=self.model_name, messages=[self._as_shown(m) for m in openai_messages], tools=tools,
                temperature=0.001, extra_body={"usage": {"include": True}, **self.extra_body})
        # ToolSandbox executes and scores original names; the history keeps what the model wrote.
        for tc in response.choices[0].message.tool_calls or []:
            # ToolSandbox turns the id into a Python variable name; some providers return bare hex
            # ids starting with a digit (Qwen) or ids with hyphens (Gemma), both SyntaxErrors.
            if not tc.id.isidentifier():
                tc.id = re.sub(r"\W", "_", tc.id)
                tc.id = tc.id if tc.id.isidentifier() else "call_" + tc.id
            self.raw_calls[tc.id] = (tc.function.name, tc.function.arguments)
            tc.function.name, tc.function.arguments = restore_call(tc.function.name, tc.function.arguments, self.maps)
        return response

    def _as_shown(self, message: dict) -> dict:
        """ToolSandbox rebuilds the history from executed code, i.e. with original names.
        Put back the names the model actually used so its own history is consistent."""
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**tc, "function": dict(zip(("name", "arguments"), self.raw_calls.get(
                    tc["id"], (tc["function"]["name"], tc["function"]["arguments"]))))}
                for tc in message["tool_calls"]]
        if message.get("role") == "tool" and message.get("tool_call_id") in self.raw_calls:
            message["name"] = self.raw_calls[message["tool_call_id"]][0]
        return message


class User(OpenAIAPIUser):
    """ToolSandbox's GPT-4o user simulator, reached through OpenRouter."""

    def __init__(self, model_name: str = USER_MODEL):
        self.model_name, self.calls = model_name, []
        self.openai_client = recording_client(self.calls)

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def model_inference(self, openai_messages, openai_tools):
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(
                model=self.model_name, messages=openai_messages, tools=openai_tools,
                extra_body={"usage": {"include": True}})


# ---- scenarios -----------------------------------------------------------------------

def scenarios(limit: int | None = None) -> dict:
    """Name -> scenario: base scenarios that need no RapidAPI tool, in their `_all_tools` variant."""
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    random.seed(42)  # ToolSandbox's CLI seeds the tool-order shuffles the same way
    all_scenarios = named_scenarios(ToolBackend.DEFAULT)
    base = sorted(name for name, s in all_scenarios.items()
                  if "_distraction_tools" not in name and not name.endswith("_all_tools")
                  and not set(s.starting_context.tool_allow_list) & RAPID_TOOLS)
    out = {}
    for name in base[:limit]:
        scenario = all_scenarios[name + "_all_tools"]
        scenario.starting_context.tool_deny_list = (scenario.starting_context.tool_deny_list or []) + sorted(RAPID_TOOLS)
        out[name] = scenario
    return out


def registry_dir(model_id: str, condition: str) -> str:
    return f"{model_id.replace('/', '_')}-{condition}"


_SCENARIOS = None


def run_one(job: tuple) -> dict:
    """Play and score one scenario for one model x condition (runs in a worker process)."""
    global _SCENARIOS
    name, model_id, condition, user_model, run_dir = job
    # ToolSandbox writes three files per scenario under <output_directory>/trajectories/<name>/; they are
    # folded into the single trajectory JSON below and the scratch directory is removed.
    tmp_dir = run_dir / ".toolsandbox_tmp" / registry_dir(model_id, condition)
    if _SCENARIOS is None:
        _SCENARIOS = scenarios()
    scenario = _SCENARIOS[name]
    agent, user = Agent(model_id, condition), User(user_model)
    roles = {RoleType.USER: user, RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(), RoleType.AGENT: agent}
    started_at, t0, error = now_iso(), time.time(), None
    try:
        played = scenario.play_and_evaluate(roles=roles, output_directory=tmp_dir, scenario_name=name)
        ev = played.evaluation_result
        result = {"similarity": ev.similarity, "milestone_similarity": ev.milestone_similarity,
                  "minefield_similarity": ev.minefield_similarity, "turn_count": ev.turn_count,
                  "milestone_mapping": ev.milestone_mapping, "minefield_mapping": ev.minefield_mapping}
    except Exception as e:  # scored 0, exactly as ToolSandbox's own CLI does
        error = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        result = {"similarity": 0, "milestone_similarity": 0, "minefield_similarity": 0,
                  "turn_count": scenario.max_messages, "milestone_mapping": {}, "minefield_mapping": {}}
    finally:
        for role in roles.values():
            role.teardown()
    scratch = tmp_dir / "trajectories" / name
    extras = {
        "conversation": _json_or_none(scratch / "conversation.json"),
        "execution_context": _json_or_none(scratch / "execution_context.json"),
        "pretty_print": (scratch / "pretty_print.txt").read_text() if (scratch / "pretty_print.txt").exists() else None,
    }
    shutil.rmtree(scratch, ignore_errors=True)
    doc = {
        "id": name, "benchmark": "toolsandbox", "model": model_id, "condition": condition,
        "user_model": user_model, "extra_body": agent.extra_body,
        "categories": [str(c) for c in scenario.categories],
        "started_at": started_at, "finished_at": now_iso(), "wall_time_s": time.time() - t0,
        "totals": totals(agent.calls), "user_totals": totals(user.calls),
        "error": error, "result": result,
        "tools_sent": agent.tools_sent, "name_maps": agent.maps,
        "api_calls": agent.calls, "user_api_calls": user.calls,
        "toolsandbox": extras,
    }
    path = run_dir / "trajectories" / registry_dir(model_id, condition) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
    return {"id": name, "model": model_id, "condition": condition, "similarity": result["similarity"],
            "error": error and error["type"]}


def _json_or_none(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ---- main ------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "index"])
    p.add_argument("--models", nargs="+", default=BASE_MODELS, choices=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--jobs", type=int, default=8, help="scenarios played in parallel (processes)")
    p.add_argument("--limit", type=int, help="only the first K scenarios (smoke test)")
    p.add_argument("--user-model", default=USER_MODEL)
    p.add_argument("--run-name", default="main", help="results go to runs/<run-name>/trajectories/")
    args = p.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing from .env"
    os.environ["TQDM_DISABLE"] = "1"  # one progress bar per worker is unreadable
    run_dir = PROJECT_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        names = list(scenarios(args.limit))
        jobs = [(n, m, c, args.user_model, run_dir)
                for m in args.models for c in args.conditions for n in names
                if not (run_dir / "trajectories" / registry_dir(m, c) / f"{n}.json").exists()]
        print(f"{len(jobs)} scenarios to play ({len(names)} per model x condition; done ones are skipped)", flush=True)
        done = 0
        with get_context("spawn").Pool(min(args.jobs, max(len(jobs), 1))) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                done += 1
                print(f"[{done}/{len(jobs)} {time.strftime('%H:%M:%S')}] {r['model']} {r['condition']} {r['id']}: "
                      f"similarity {r['similarity']:.2f}{' ERROR ' + r['error'] if r['error'] else ''}", flush=True)
        shutil.rmtree(run_dir / ".toolsandbox_tmp", ignore_errors=True)
    build_index(run_dir)


if __name__ == "__main__":
    main()
