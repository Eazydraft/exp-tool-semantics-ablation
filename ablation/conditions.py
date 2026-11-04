"""Documentation ablation conditions (paper, Section 4.2).

`apply_condition` takes BFCL's original function docs for one task and returns
  (a) the ablated docs that are shown to the model, and
  (b) name maps so the model's calls can be translated back to the original
      names before BFCL executes and scores them (`restore_call`).

Only documentation is touched: parameter types, required/optional, enums,
defaults and the response schema structure are always preserved.
"""

import copy
import json

# Which documentation components are kept ORIGINAL (True) vs ablated (False).
CONDITIONS = {
    "D0": dict(tool_names=True, tool_desc=True, param_names=True, param_desc=True),
    "D1": dict(tool_names=True, tool_desc=False, param_names=True, param_desc=True),
    "D2": dict(tool_names=True, tool_desc=True, param_names=True, param_desc=False),
    "D3": dict(tool_names=False, tool_desc=True, param_names=True, param_desc=True),
    "D4": dict(tool_names=True, tool_desc=True, param_names=False, param_desc=True),
    "D5": dict(tool_names=False, tool_desc=False, param_names=False, param_desc=False),
    # Follow-ups to D2: what if the information the descriptions carried
    # is expressed structurally instead (D6), or kept only for the six parameters that
    # accounted for most D2 errors (D7)?
    "D6": dict(tool_names=True, tool_desc=True, param_names=True, param_desc=False, structure=True),
    "D7": dict(tool_names=True, tool_desc=True, param_names=True, param_desc=False, keep_desc=True),
    # D8 removes parameter names and parameter descriptions together (tool names and descriptions
    # kept): the direct test of how far those two carriers restate each other.
    "D8": dict(tool_names=True, tool_desc=True, param_names=False, param_desc=False),
}

# The six parameters whose values went wrong most in D2 (paper, Table 5).
SIX_PARAMS = {("place_order", "order_type"), ("startEngine", "ignitionMode"), ("estimate_distance", "cityA"),
              ("estimate_distance", "cityB"), ("post_tweet", "tags"), ("post_tweet", "mentions")}
# D6: structural replacements for information that BFCL keeps only in prose.
RENAMES = {("estimate_distance", "cityA"): "zipcodeA", ("estimate_distance", "cityB"): "zipcodeB"}
ITEM_PATTERNS = {("post_tweet", "tags"): "^#", ("post_tweet", "mentions"): "^@"}
PROSE_ENUMS = {("place_order", "order_type"): ["Buy", "Sell"]}  # written "(Buy/Sell)" rather than "[Enum]: ..."


def apply_condition(functions: list[dict], condition_id: str) -> tuple[list[dict], dict]:
    cfg = CONDITIONS[condition_id]
    functions = copy.deepcopy(functions)
    tool_map = {}  # generic tool name -> original tool name
    param_map = {}  # tool name shown to the model -> nested rename map (see _rename_properties)

    for i, fn in enumerate(functions):
        if not cfg["tool_names"]:
            original = fn["name"]
            fn["name"] = f"tool_{i + 1:02d}"  # e.g. search_files -> tool_01
            tool_map[fn["name"]] = original

        if not cfg["tool_desc"]:
            fn.pop("description", None)
            # The return-value docs describe what the tool does, so they go with the tool description.
            _strip_descriptions(fn.get("response", {}), blank=False)

        params = fn["parameters"]
        if not cfg["param_names"]:
            param_map[fn["name"]] = _rename_properties(params)

        if cfg.get("structure"):
            param_map[fn["name"]] = _add_structure(original_name(fn, tool_map), params)

        if not cfg["param_desc"]:
            keep = {p for t, p in SIX_PARAMS if t == fn["name"]} if cfg.get("keep_desc") else set()
            # Blank instead of delete: BFCL's convert_to_tool appends a type note to
            # the description of float parameters and expects the key to exist.
            _strip_descriptions(params, blank=True, keep=keep)

    return functions, {"tool": tool_map, "param": param_map}


def original_name(fn: dict, tool_map: dict) -> str:
    return tool_map.get(fn["name"], fn["name"])


def _add_structure(tool: str, params: dict) -> dict:
    """D6: move prose-only constraints into the schema. Enum lists written as "[Enum]: ..." in a
    description become `enum`; two zip-code parameters get names that say so; tags/mentions
    get an item `pattern`. Returns the param-name map `_restore_value` expects."""
    props = params["properties"]
    pmap = {}
    for p in list(props):
        v = props[p]
        desc = v.get("description", "")
        if "enum" not in v:
            if (tool, p) in PROSE_ENUMS:
                v["enum"] = PROSE_ENUMS[(tool, p)]
            elif "[Enum]:" in desc:
                raw = desc.split("[Enum]:", 1)[1].strip().rstrip(".")
                vals = [x.strip().strip("[]\"' ") for x in raw.strip("[]").split(",")]
                if v.get("type") == "array" and "items" in v:
                    v["items"]["enum"] = vals
                else:
                    v["enum"] = vals
        if (tool, p) in ITEM_PATTERNS and "items" in v:
            v["items"]["pattern"] = ITEM_PATTERNS[(tool, p)]
        if (tool, p) in RENAMES:
            new = RENAMES[(tool, p)]
            props[new] = props.pop(p)
            params["required"] = [new if r == p else r for r in params.get("required", [])]
            pmap[new] = {"name": p, "nested": {}}
    return pmap


def _rename_properties(schema: dict) -> dict:
    """Rename the `properties` of a JSON-schema object node to arg_1..arg_n, in place,
    recursing into nested objects and arrays of objects.

    Returns {generic_name: {"name": original_name, "nested": map for that property's value}},
    which `_restore_value` uses to translate the model's argument values back.
    """
    props = schema.get("properties")
    if not props:
        return {}
    renames = {p: f"arg_{j + 1}" for j, p in enumerate(props)}  # e.g. directory_path -> arg_1
    schema["properties"] = {renames[p]: v for p, v in props.items()}
    if "required" in schema:
        schema["required"] = [renames[p] for p in schema["required"]]
    return {g: {"name": o, "nested": _rename_value_schema(schema["properties"][g])}
            for o, g in renames.items()}


def _rename_value_schema(prop: dict) -> dict:
    """Map for the value of one property: an object has `properties`, an array has `items`."""
    if "items" in prop:
        return {"items": _rename_value_schema(prop["items"])}
    return _rename_properties(prop)


def _restore_value(value, pmap: dict):
    """Translate generic keys inside a model-supplied argument value back to the original keys."""
    if not pmap:
        return value
    if isinstance(value, dict):
        return {pmap.get(k, {}).get("name", k): _restore_value(v, pmap.get(k, {}).get("nested"))
                for k, v in value.items()}
    if isinstance(value, list) and "items" in pmap:
        return [_restore_value(v, pmap["items"]) for v in value]
    return value


def _strip_descriptions(schema: dict, blank: bool, keep: set = frozenset()) -> None:
    """Recursively remove natural-language `description` fields from a JSON schema,
    except for the top-level properties named in `keep`."""
    if not isinstance(schema, dict):
        return
    if "description" in schema:
        if blank:
            schema["description"] = ""
        else:
            del schema["description"]
    for name, prop in schema.get("properties", {}).items():
        if name not in keep:
            _strip_descriptions(prop, blank)
    _strip_descriptions(schema.get("items"), blank)


def restore_call(name: str, arguments_json: str, maps: dict) -> tuple[str, str]:
    """Map one model tool call (as shown to the model) back to BFCL's original names.

    Names the model made up (not in the maps) are left untouched, so invalid calls
    fail inside BFCL's executor exactly as they would in the unmodified benchmark.
    """
    original_name = maps["tool"].get(name, name)
    try:
        args = json.loads(arguments_json)
    except (TypeError, ValueError):
        return original_name, arguments_json  # malformed JSON: let BFCL's decoder report it
    args = _restore_value(args, maps["param"].get(name, {}))
    return original_name, json.dumps(args)


def registry_name(model_id: str, condition_id: str) -> str:
    """BFCL registry key for one model x condition, e.g. 'openai/gpt-6-luna-D1'.

    BFCL turns '/' into '_' for directory names and back again when evaluating,
    so the model id itself must not contain '_'.
    """
    assert "_" not in model_id, f"model id must not contain '_': {model_id}"
    return f"{model_id}-{condition_id}"


def split_registry_name(name: str) -> tuple[str, str]:
    model_id, condition_id = name.rsplit("-", 1)
    return model_id, condition_id
