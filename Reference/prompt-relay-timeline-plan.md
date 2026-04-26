# RSPromptRelayTimeline — Build Plan

A canvas-based timeline node that outputs the relay JSON consumed by
`RSPromptRelayEncode`. Sits alongside (not inside) the encoder so hand-authored
JSON still works.

## Why a separate node

- Decouples authoring (timeline / JSON / Ollama / paste) from encoding.
- Encoder `RSPromptRelayEncode` already accepts any JSON STRING in its `prompt`
  field — zero changes needed.
- Workflow save/restore comes free: the timeline state lives in a STRING widget,
  ComfyUI persists it automatically.

## Architecture

```
[RSPromptRelayTimeline]  ──STRING(JSON)──► [RSPromptRelayEncode.prompt] ─► CONDITIONING
        ↑
   canvas widget (web/prompt_relay_timeline.js)
```

### Python — `nodes/prompt_relay_timeline.py`

```python
class RSPromptRelayTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_duration_sec": ("FLOAT", {"default": 4.0,  "min": 0.1, "max": 600.0, "step": 0.01}),
                "frame_rate":         ("FLOAT", {"default": 25.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "timeline_data":      ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relay_json",)
    FUNCTION = "build"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = False
    def build(self, total_duration_sec, frame_rate, timeline_data):
        # Pass-through: the JS widget keeps `timeline_data` populated with
        # canonical JSON. If empty, emit an empty object (encoder degrades to
        # single-prompt path).
        s = (timeline_data or "").strip() or "{}"
        return (s,)
```

That's the entire Python side. The JS widget is responsible for keeping
`timeline_data` valid JSON matching the encoder schema.

### JS — `web/prompt_relay_timeline.js`

Registers an extension via `app.registerExtension`, hooks
`beforeRegisterNodeDef` for `RSPromptRelayTimeline`, and overrides the default
rendering of the `timeline_data` widget.

Approach (vanilla, no framework):
1. On `onNodeCreated`, find the `timeline_data` widget. Keep it (so the value
   round-trips with the workflow JSON) but hide its default DOM via
   `widget.computeSize = () => [0, -4]`.
2. Add a custom widget via `this.addCustomWidget({ type: "RS_TIMELINE", … })`
   that renders the canvas during `onDrawForeground` — or, simpler for v1,
   create an HTMLCanvasElement and inject it as a DOM widget via
   `this.addDOMWidget("timeline", "timeline", canvas)`.
3. State held in a JS object `{ global, segments: [{ id, t_start, t_end, prompt }] }`,
   serialized to `timeline_data.value` on every change. Reverse: parse on
   `onConfigure` (workflow load).
4. Read `total_duration_sec` and `frame_rate` widgets to compute pixel↔seconds
   scaling and snap-to-frame.

## v1 features

| Feature | Behavior |
|---|---|
| Time axis | Horizontal, scaled to `total_duration_sec` |
| Tick marks | Every 1 sec (long), every 1/fps (short, snap target) |
| Global prompt | Editable text input above the canvas |
| Segment block | Rectangle on its own row; label = first ~30 chars of prompt |
| Add segment | Button "Add segment" — appended at end of last segment (or 0–1s) |
| Move | Drag body horizontally; clamp to `[0, total_duration]` |
| Resize | Drag left/right edge handle (8px hot zone); enforce `t_end > t_start` |
| Edit prompt | Double-click → inline `<textarea>` overlay that disappears on Enter / blur |
| Delete | Right-click → "Delete", or `Del` key when selected |
| Snap | Hold Shift to disable; default snap = 1/`frame_rate` |
| Selection | Click selects (highlight); only one segment selected at a time in v1 |

## v1 NON-goals

- Undo / redo
- Multi-select / box-drag-select
- Copy / paste / clipboard sync
- Stacking multiple segments on the same row when they overlap (v1 uses one row
  per segment in declaration order; overlapping segments simply share the
  visible time range across different rows)
- Audio waveform display
- Per-segment color customization

## JSON schema (matches encoder contract)

```json
{
  "global": "string",
  "segments": [
    {"t_start": 0.0, "t_end": 1.5, "prompt": "..."},
    {"t_start": 1.5, "t_end": 3.0, "prompt": "..."}
  ]
}
```

The JS serializer rounds `t_start` / `t_end` to 3 decimal places and writes
segments in row order (which is also creation order until v2 adds reordering).

## Wire-up

1. Add to root `__init__.py` mappings:
   - `"RSPromptRelayTimeline": RSPromptRelayTimeline`
   - Display: `"RS Prompt Relay Timeline"`
2. `web/prompt_relay_timeline.js` is auto-discovered via `WEB_DIRECTORY = "./web"`
   already declared in `__init__.py`.
3. Workflow `RSPromptRelayTimeline → RSPromptRelayEncode (prompt) → RSLTXVGenerate`.

## Build order

| Step | Deliverable |
|---|---|
| **G1** | Python skeleton + node registration. Ship empty JS file (canvas-less). Wire renders default STRING textarea — confirm node appears in ComfyUI. |
| **G2** | Minimal canvas (read-only) — render time axis + tick marks + segments parsed from JSON. No interaction yet. |
| **G3** | Add segment button + drag-to-move + drag-edges-to-resize. Frame snap. JSON serializes on change. |
| **G4** | Edit prompt (inline textarea). Delete (right-click / key). Selection highlight. |
| **G5** | Global prompt input. Polish: cursor cues, hover highlight, num_frames display. |

Each step is a commit. G1+G2 should land same day; G3 is the meaty one.

## Risks

- ComfyUI's frontend churn — `addDOMWidget` API has changed across versions.
  Verify against the user's current ComfyUI install (E:/ComfyUI is git head).
- Canvas vs DOM trade: DOM is easier for inline edit (just place a textarea) but
  pixel-precise drag is cleaner on canvas. v1 uses canvas with a transient DOM
  textarea overlay during edit.
- High-DPI displays: respect `devicePixelRatio` in canvas sizing.
