# Server-side UI workflow → API prompt converter for rs-studio.
#
# ComfyUI's frontend has graphToPrompt() in JavaScript that converts a
# UI-format workflow (the one "Save" produces) into the API prompt
# /prompt expects. There's no equivalent on the Python side, so any
# external tool that wants to drive runs from a UI workflow has to
# reimplement the conversion — and stay in sync with every quirk of
# ComfyUI's widget system.
#
# This endpoint runs the conversion ON the pod, where it has direct
# access to NODE_CLASS_MAPPINGS. That means the canonical widget
# ordering for any installed node is authoritative — no positional
# guessing from the workflow JSON, no need for clients to hardcode
# auto-widget patterns like control_after_generate.
#
# POST /rs/uitoapi
#   request body: the raw UI workflow JSON (the same shape as a
#                 saved .json from ComfyUI's File → Save menu)
#   response:     { "prompt": <api prompt>, "warnings": [...] }
#
# Backward-compatible: rs-studio falls back to its local converter
# when this endpoint isn't available (older rs-nodes on the pod).

from aiohttp import web

import server
import nodes


# Frontend-only nodes that have no class_type and shouldn't appear
# in the API prompt. Mirrors workflow-converter.ts's SKIP_TYPES list.
SKIP_TYPES = {
    "Note",
    "PrimitiveNode",
    "Reroute",
    "GroupHeader",
    "MarkdownNote",
}

# Connection-typed inputs — never widgets, value comes from a link.
CONNECTION_TYPES = {
    "MODEL",
    "CLIP",
    "VAE",
    "CONDITIONING",
    "LATENT",
    "IMAGE",
    "MASK",
    "CONTROL_NET",
    "STYLE_MODEL",
    "CLIP_VISION",
    "CLIP_VISION_OUTPUT",
    "GLIGEN",
    "SAMPLER",
    "SIGMAS",
    "NOISE",
    "GUIDER",
    "AUDIO",
    "VIDEO",
}


def _is_widget_type(declared):
    """True when the declared type is a widget (not a connection slot)."""
    if isinstance(declared, list):
        # Enum: ["option_a", "option_b", ...] — always a widget.
        return True
    if isinstance(declared, str):
        if declared in CONNECTION_TYPES:
            return False
        # Widget primitive (INT / FLOAT / STRING / BOOLEAN / etc.)
        return True
    return False


def _has_auto_control_after_generate(input_name):
    """ComfyUI's frontend auto-injects a control_after_generate combo
    widget after any widget named seed / noise_seed. The auto-widget is
    NOT in the node class's INPUT_TYPES but IS serialized into
    widgets_values, so the cursor must skip past it."""
    return input_name in ("seed", "noise_seed")


def _canonical_input_order(node_class):
    """Return [(name, declared_type), ...] in the canonical order
    ComfyUI's frontend uses to lay out widgets / connections.

    Required first, then optional. Hidden inputs are excluded — they
    don't appear in widgets_values and aren't user-facing.
    """
    try:
        spec = node_class.INPUT_TYPES()
    except Exception:
        return None
    order = []
    for section in ("required", "optional"):
        section_dict = spec.get(section, {}) or {}
        for name, decl in section_dict.items():
            # decl is usually a tuple/list: (TYPE, {options...}) or just (TYPE,)
            declared_type = decl[0] if isinstance(decl, (list, tuple)) and decl else None
            order.append((name, declared_type))
    return order


def _index_links_by_dest(workflow):
    """Map (dest_node_id, dest_slot) -> (src_node_id, src_slot)."""
    by_dest = {}
    for link in workflow.get("links", []) or []:
        if not isinstance(link, (list, tuple)) or len(link) < 6:
            continue
        # ComfyUI link tuple: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        _, src_node, src_slot, dst_node, dst_slot, _ = link[:6]
        by_dest[(dst_node, dst_slot)] = (src_node, src_slot)
    return by_dest


def _convert_node(node, link_by_dest, warnings):
    """Convert one workflow node to its API prompt entry. Returns
    (key, value) or None when the node should be omitted."""
    node_id = node.get("id")
    node_type = node.get("type")
    if node_id is None or not node_type:
        return None
    if node_type in SKIP_TYPES:
        return None
    # mode 2 = muted, 4 = bypass; we skip both. Real bypass should
    # passthrough the wires, but that's a layer of complexity we
    # don't need yet — same behavior as workflow-converter.ts.
    if node.get("mode") in (2, 4):
        return None

    widgets_values = node.get("widgets_values") or []
    workflow_inputs = node.get("inputs") or []

    api_inputs = {}
    node_class = nodes.NODE_CLASS_MAPPINGS.get(node_type)

    # Resolve every connection input first — these come straight from
    # the link table by destination slot index, regardless of widget
    # arithmetic. We use the workflow's inputs[] order for slot indices
    # because that's what links reference.
    inputs_by_name = {}
    for slot_idx, inp in enumerate(workflow_inputs):
        name = inp.get("name")
        if not name:
            continue
        inputs_by_name[name] = (slot_idx, inp)
        link_id = inp.get("link")
        if link_id is not None:
            src = link_by_dest.get((node_id, slot_idx))
            if src is not None:
                api_inputs[name] = [str(src[0]), int(src[1])]

    # Widget values: iterate the CANONICAL order (NODE_CLASS_MAPPINGS),
    # falling back to the workflow's inputs[] ordering when the node
    # class isn't installed (custom node we don't have on this pod).
    canonical = _canonical_input_order(node_class) if node_class else None
    if canonical is None:
        # Fallback: derive widget order from workflow inputs that have
        # a `widget` field. Less reliable but only fires for unknown
        # custom nodes.
        canonical = []
        for inp in workflow_inputs:
            name = inp.get("name")
            if not name:
                continue
            decl_type = inp.get("type")
            if inp.get("widget") is not None or _is_widget_type(decl_type):
                canonical.append((name, decl_type))
            else:
                canonical.append((name, decl_type))  # connection — preserved for ordering, no widget value

    # Walk canonical inputs, mapping widget-typed entries to
    # widgets_values positionally. Widgets that have been "promoted"
    # to inputs (link != None) still occupy their widgets_values slot
    # in the saved workflow, so the cursor advances either way.
    cursor = 0
    for name, declared_type in canonical:
        if not _is_widget_type(declared_type):
            # Connection-only input — already resolved above (or not),
            # nothing to do here. Cursor doesn't move.
            continue

        slot, inp = inputs_by_name.get(name, (None, None))
        is_promoted = inp is not None and inp.get("link") is not None
        if not is_promoted:
            # Real widget — read its value from widgets_values.
            if cursor < len(widgets_values):
                api_inputs[name] = widgets_values[cursor]
            else:
                warnings.append(
                    f"node {node_id} ({node_type}): widget '{name}' "
                    f"requested cursor {cursor} but widgets_values has "
                    f"only {len(widgets_values)} entries"
                )
        # cursor advances whether the slot is real or promoted —
        # ComfyUI keeps the slot in widgets_values either way.
        cursor += 1
        # Auto-injected control_after_generate sits after seed widgets
        # in widgets_values without a corresponding INPUT_TYPES entry.
        if _has_auto_control_after_generate(name):
            cursor += 1

    return str(node_id), {
        "class_type": node_type,
        "inputs": api_inputs,
    }


def convert_ui_to_api(workflow):
    """Convert a UI-format workflow JSON (the saved Comfy graph) into
    the API prompt /prompt expects."""
    warnings = []
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a JSON object")
    nodes_list = workflow.get("nodes")
    if not isinstance(nodes_list, list):
        raise ValueError("workflow.nodes is missing or not a list")

    link_by_dest = _index_links_by_dest(workflow)
    api_prompt = {}

    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        result = _convert_node(node, link_by_dest, warnings)
        if result is None:
            continue
        key, value = result
        api_prompt[key] = value

    return api_prompt, warnings


# Register on the running ComfyUI server. Tolerates being imported in a
# context where PromptServer isn't initialised yet (e.g. unit-test
# import of rs-nodes) by guarding against attribute errors.
try:
    _server = server.PromptServer.instance
except Exception:
    _server = None

if _server is not None:

    @_server.routes.post("/rs/uitoapi")
    async def _ui_to_api_route(request):
        try:
            workflow = await request.json()
        except Exception as err:
            return web.json_response(
                {"error": f"invalid JSON body: {err}"}, status=400
            )
        try:
            api_prompt, warnings = convert_ui_to_api(workflow)
        except ValueError as err:
            return web.json_response({"error": str(err)}, status=400)
        except Exception as err:
            # Log full traceback server-side, return a short message to
            # the client — the renderer doesn't need a stack.
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": f"conversion failed: {err}"}, status=500
            )
        return web.json_response({"prompt": api_prompt, "warnings": warnings})
