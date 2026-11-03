"""BFCL model handler for the ablation study.

Subclasses BFCL's own OpenAI-compatible handler so the multi-turn loop, the tool
execution environment and the scoring are reused untouched. It only:
  1. talks to OpenRouter instead of OpenAI,
  2. swaps in the ablated tool schemas (conditions.py) and maps the model's calls
     back to the original names before BFCL executes them,
  3. records every raw HTTP request/response and writes one JSON file per trajectory.

BFCL runs each task start-to-finish on a single worker thread, so all per-task
state (name maps, captured HTTP calls) lives in a thread-local.
"""

import datetime
import json
import os
import threading
import time
import traceback
from pathlib import Path

import httpx
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.utils import convert_to_tool, retry_with_backoff
from bfcl_eval.utils import make_json_serializable
from openai import APIConnectionError, InternalServerError, RateLimitError
from tenacity import stop_after_attempt

from ablation.conditions import apply_condition, restore_call, split_registry_name
from ablation.models import MODELS

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 180



def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AblationHandler(OpenAICompletionsHandler):
    # Set by run.py before generation starts.
    trajectory_dir: Path = None
    run_info: dict = {}

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        self._tls = threading.local()
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_id, self.condition = split_registry_name(registry_name)
        self.extra_body = MODELS[self.model_id].get("extra_body", {})  # e.g. reasoning on/off for the controlled pairs

    # ---- OpenRouter client with raw HTTP capture -------------------------------

    def _build_client_kwargs(self):
        http_client = httpx.Client(
            event_hooks={"request": [self._on_request], "response": [self._on_response]}
        )
        return dict(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            http_client=http_client,
            # A completion normally takes seconds; a provider that never answers would otherwise
            # hold a worker for 10 min x SDK retries x our retries. Bound it tightly instead.
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10),
            max_retries=1,
        )

    def _on_request(self, request: httpx.Request):
        calls = getattr(self._tls, "calls", None)
        if calls is None:
            return
        headers = {k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in request.headers.items()}
        calls.append({
            "request": {
                "sent_at": now_iso(),
                "method": request.method,
                "url": str(request.url),
                "headers": headers,
                "body": _json_or_text(request.content),
            }
        })

    def _on_response(self, response: httpx.Response):
        calls = getattr(self._tls, "calls", None)
        if not calls:
            return
        response.read()
        calls[-1]["response"] = {
            "received_at": now_iso(),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": _json_or_text(response.content),
        }
        calls[-1]["latency_s"] = response.elapsed.total_seconds()

    @retry_with_backoff(
        error_type=[RateLimitError, APIConnectionError, InternalServerError],
        stop=stop_after_attempt(4),  # worst case per hung request: 4 x 2 x 180s = 24 min
        reraise=True,
    )
    def generate_with_backoff(self, **kwargs):
        # Ask OpenRouter to report cost and token details in `usage`.
        kwargs["extra_body"] = {"usage": {"include": True}, **self.extra_body}
        start = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        return api_response, time.time() - start

    # ---- Ablation: tool schemas out, original names back in ---------------------

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions, maps = apply_condition(test_entry["function"], self.condition)
        inference_data["tools"] = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        self._tls.maps = maps
        self._tls.tools_sent = inference_data["tools"]
        return inference_data

    def _parse_query_response_FC(self, api_response) -> dict:
        data = super()._parse_query_response_FC(api_response)
        # `model_responses` is what BFCL executes and scores: translate back to original
        # names. The chat history keeps the message exactly as the model produced it.
        if isinstance(data["model_responses"], list):
            data["model_responses"] = [
                dict([restore_call(name, args, self._tls.maps)])
                for call in data["model_responses"]
                for name, args in call.items()
            ]
        # OpenRouter exposes reasoning as `message.reasoning`; BFCL logs it if present.
        reasoning = getattr(api_response.choices[0].message, "reasoning", None)
        if reasoning:
            data["reasoning_content"] = reasoning
        return data

    # ---- Per-trajectory JSON dump -----------------------------------------------

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        self._tls.calls, self._tls.maps, self._tls.tools_sent = [], None, None
        started_at, t0 = now_iso(), time.time()
        result, metadata, error = None, {}, None
        try:
            result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
        except Exception as e:  # BFCL records this as an inference error; we still keep the trajectory
            error = {"message": str(e), "traceback": traceback.format_exc()}
            raised = e
        self._write_trajectory(test_entry, result, metadata, started_at, time.time() - t0, error)
        if error:
            raise raised
        return result, metadata

    def _write_trajectory(self, test_entry, result, metadata, started_at, wall_time_s, error):
        calls = self._tls.calls
        usages = [c["response"]["body"].get("usage") or {} for c in calls
                  if isinstance(c.get("response", {}).get("body"), dict)]
        totals = {
            "api_calls": len(calls),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
            "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usages),
            "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usages),
            "cost_usd": sum(u.get("cost", 0) or 0 for u in usages),
            "latency_s": sum(c.get("latency_s", 0) for c in calls),
        }
        doc = {
            "id": test_entry["id"],
            "model": self.model_id,
            "condition": self.condition,
            "registry_name": self.registry_name,
            "extra_body": self.extra_body,
            "run": self.run_info,
            "started_at": started_at,
            "finished_at": now_iso(),
            "wall_time_s": wall_time_s,
            "totals": totals,
            "error": error,
            # The task as BFCL loaded it (minus the 128-function doc list, which is in `tools_sent` in ablated form).
            "task": {k: v for k, v in test_entry.items() if k != "function"},
            "tools_sent": self._tls.tools_sent,
            "name_maps": self._tls.maps,
            # What BFCL writes to its own result file: per-turn, per-step tool calls (original names) + logs.
            "bfcl_result": result,
            "bfcl_metadata": metadata,
            # Every raw HTTP exchange with OpenRouter, in order (including retried attempts).
            "api_calls": calls,
        }
        out_dir = Path(self.trajectory_dir) / self.registry_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{test_entry['id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(make_json_serializable(doc), ensure_ascii=False, indent=1))
        tmp.replace(path)


def _json_or_text(content: bytes):
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content.decode("utf-8", errors="replace")
