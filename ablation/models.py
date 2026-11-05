"""The models under study, keyed by the id used in registry names and result directories.

OpenRouter prices (USD per 1M tokens, as listed on 2026-11-02) only feed BFCL's leaderboard CSV;
the actual cost of every call is recorded from OpenRouter's usage field.

The *-nothink / *-think entries are reasoning-controlled pairs of a base model: the same model id
with reasoning explicitly switched off (GPT-6 Luna and Claude Haiku 5.5 reason by default) or on
(Gemma 4 does not). `extra_body` is merged into every request. Qwen3-30B-A3B-Instruct-2507 has
no reasoning mode on OpenRouter (the thinking variant is a different checkpoint), so it has no pair.

The *-rerun entry is the same model with the same settings launched a second time, so that a
repeated run sits next to the first one in the same run directory (run-to-run noise floor).
"""

MODELS = {
    "openai/gpt-6-luna": dict(org="OpenAI", input_price=0.10, output_price=0.50),
    "anthropic/claude-haiku-5.5": dict(org="Anthropic", input_price=0.10, output_price=0.50),
    "qwen/qwen3-30b-a3b-instruct-2507": dict(org="Qwen", input_price=0.10, output_price=0.30),
    "google/gemma-4-26b-a4b-it": dict(org="Google", input_price=0.064, output_price=0.191),
    "openai/gpt-6-luna-nothink": dict(
        org="OpenAI", model="openai/gpt-6-luna", input_price=0.10, output_price=0.50,
        extra_body={"reasoning": {"enabled": False}}),
    "anthropic/claude-haiku-5.5-nothink": dict(
        org="Anthropic", model="anthropic/claude-haiku-5.5", input_price=0.10, output_price=0.50,
        extra_body={"reasoning": {"enabled": False}}),
    # The main run's Gemma requests went to DeepInfra; prefer it here too so the pair differs only in reasoning.
    "google/gemma-4-26b-a4b-it-think": dict(
        org="Google", model="google/gemma-4-26b-a4b-it", input_price=0.064, output_price=0.191,
        extra_body={"reasoning": {"enabled": True}, "provider": {"order": ["DeepInfra"]}}),
    "anthropic/claude-haiku-5.5-rerun": dict(org="Anthropic", model="anthropic/claude-haiku-5.5", input_price=0.10, output_price=0.50),
    "openai/gpt-6-luna-rerun": dict(org="OpenAI", model="openai/gpt-6-luna", input_price=0.10, output_price=0.50),
    "qwen/qwen3-30b-a3b-instruct-2507-rerun": dict(org="Qwen", model="qwen/qwen3-30b-a3b-instruct-2507", input_price=0.10, output_price=0.30),
    "google/gemma-4-26b-a4b-it-rerun": dict(org="Google", model="google/gemma-4-26b-a4b-it", input_price=0.064, output_price=0.191),
}

BASE_MODELS = [m for m, info in MODELS.items() if "model" not in info]


def api_model(model_id: str) -> str:
    """The id sent to OpenRouter (the base model for a reasoning variant)."""
    return MODELS[model_id].get("model", model_id)
