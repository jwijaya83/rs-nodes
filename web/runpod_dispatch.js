// RSRunOnRunPod — "Run on RunPod" button + live status panel.
//
// The dispatcher node carries a hidden _workflow_json input that the
// backend uses as the captured workflow to send to the pod. Users
// shouldn't queue this node via the normal Queue Prompt button (that
// would submit the whole graph locally and OOM on the heavy nodes).
// Instead, this extension:
//
//   1. Hides the _workflow_json widget so users can't accidentally
//      edit it.
//   2. Adds a "Run on RunPod" button. When clicked, it:
//        a. Calls app.graphToPrompt() to materialise the full local
//           graph as a ComfyUI API-format prompt.
//        b. Strips the dispatcher node from that prompt and JSON-
//           encodes the rest into the dispatcher's _workflow_json
//           widget value.
//        c. Calls graphToPrompt() again so the now-updated widget
//           value is captured.
//        d. Submits ONLY the dispatcher node id from that fresh
//           prompt as a single-node queue item.
//      Net effect: the dispatcher runs locally, but everything else
//      it captured runs on the remote pod via the dispatcher's
//      Python implementation.
//   3. Renders a status panel on the node body that is fed by the
//      backend's progress events (rs.runpod.phase, rs.runpod.progress,
//      rs.runpod.log).
//
// The panel is a single multiline widget rather than a custom DOM
// element so that node serialization round-trips don't fight us.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STATUS_WIDGET_NAME = "_runpod_status";
const STATUS_MAX_LINES = 50;

// ---------------------------------------------------------------------------
// Status widget
// ---------------------------------------------------------------------------

function ensureStatusWidget(node) {
    let widget = node.widgets?.find((w) => w.name === STATUS_WIDGET_NAME);
    if (widget) return widget;

    const initial =
        "Click 'Run on RunPod' to dispatch the rest of this graph.\n" +
        "Status will appear here once the run starts.";

    // ComfyUI's STRING multiline widget has the right shape for a
    // status panel: read-only text, scrollable, sized with the node.
    widget = node.addWidget(
        "STRING",
        STATUS_WIDGET_NAME,
        initial,
        () => {},
        { multiline: true },
    );
    // Mark as a synthetic display widget — don't serialize, don't ask
    // the user for input. The leading underscore in the name doubles
    // as a visual signal.
    widget.serialize = false;
    if (widget.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = "0.85";
    }
    node.setSize(node.computeSize());
    return widget;
}

function appendStatusLine(node, line) {
    const widget = ensureStatusWidget(node);
    const stamp = new Date().toLocaleTimeString();
    const formatted = `[${stamp}] ${line}`;
    let lines = (widget.value || "").split("\n").filter((s) => s.length > 0);
    // First message after the placeholder replaces it instead of stacking.
    if (lines.length === 1 && lines[0].startsWith("Click 'Run on RunPod'"))
        lines = [];
    if (lines.length >= STATUS_MAX_LINES) {
        lines = lines.slice(lines.length - STATUS_MAX_LINES + 1);
    }
    lines.push(formatted);
    widget.value = lines.join("\n");
    if (widget.inputEl) {
        widget.inputEl.value = widget.value;
        widget.inputEl.scrollTop = widget.inputEl.scrollHeight;
    }
    node.setDirtyCanvas(true, true);
}

function setProgressLine(node, value, max, label) {
    const widget = ensureStatusWidget(node);
    const pct = max > 0 ? Math.round((value / max) * 100) : 0;
    const barWidth = 20;
    const filled = Math.max(
        0,
        Math.min(barWidth, Math.round((value / Math.max(1, max)) * barWidth)),
    );
    const bar = "[" + "#".repeat(filled) + "-".repeat(barWidth - filled) + "]";
    const line = `${bar} ${value}/${max} (${pct}%)${label ? ` — ${label}` : ""}`;

    // Replace the last line if it was already a progress line, so the
    // panel doesn't fill up with one entry per sampler step.
    let lines = (widget.value || "").split("\n");
    if (lines.length > 0 && lines[lines.length - 1].includes("[")
        && lines[lines.length - 1].includes("]")
        && /\d+\/\d+/.test(lines[lines.length - 1])) {
        lines[lines.length - 1] = line;
    } else {
        if (lines.length >= STATUS_MAX_LINES) {
            lines = lines.slice(lines.length - STATUS_MAX_LINES + 1);
        }
        lines.push(line);
    }
    widget.value = lines.join("\n");
    if (widget.inputEl) {
        widget.inputEl.value = widget.value;
        widget.inputEl.scrollTop = widget.inputEl.scrollHeight;
    }
    node.setDirtyCanvas(true, true);
}

function clearStatus(node) {
    const widget = ensureStatusWidget(node);
    widget.value = "";
    if (widget.inputEl) widget.inputEl.value = "";
}

// ---------------------------------------------------------------------------
// Hidden widget management
// ---------------------------------------------------------------------------

function hideWorkflowWidget(node) {
    const widget = node.widgets?.find((w) => w.name === "_workflow_json");
    if (!widget) return;
    // Mark as hidden so it doesn't render. computeSize() respects this.
    widget.type = "hidden";
    // Override draw so the widget area collapses.
    widget.computeSize = () => [0, -4];
    if (widget.inputEl) widget.inputEl.style.display = "none";
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

async function runOnRunPod(node) {
    try {
        clearStatus(node);
        appendStatusLine(node, "Capturing graph...");

        // Snapshot the full graph in API-prompt form. This is what
        // ComfyUI normally submits to /prompt — using its own
        // serialization avoids re-implementing widget-to-input logic.
        const captured = await app.graphToPrompt();
        const fullOutput = captured.output || {};

        const dispatcherId = String(node.id);
        if (!fullOutput[dispatcherId]) {
            throw new Error(
                "Dispatcher node missing from graph snapshot. "
                + "Try saving the workflow once and clicking again."
            );
        }

        // Build the captured-workflow blob: everything EXCEPT this dispatcher.
        const stripped = {};
        for (const [nid, nd] of Object.entries(fullOutput)) {
            if (nid === dispatcherId) continue;
            // Defensive: refuse any other RSRunOnRunPod nodes too, so
            // a graph with two dispatchers can't recurse.
            if (nd?.class_type === "RSRunOnRunPod") continue;
            stripped[nid] = nd;
        }

        if (Object.keys(stripped).length === 0) {
            throw new Error(
                "Graph contains only the dispatcher. "
                + "Add at least one other node before dispatching."
            );
        }

        // Stuff the JSON into the dispatcher's _workflow_json widget.
        const wfWidget = node.widgets.find((w) => w.name === "_workflow_json");
        if (!wfWidget) {
            throw new Error("Dispatcher node missing _workflow_json widget.");
        }
        wfWidget.value = JSON.stringify(stripped);

        appendStatusLine(
            node,
            `Captured ${Object.keys(stripped).length} node(s). Submitting...`
        );

        // Re-snapshot now that the widget value is set, so the API
        // prompt for the dispatcher carries the full captured workflow.
        const fresh = await app.graphToPrompt();
        const dispatcherEntry = fresh.output[dispatcherId];
        if (!dispatcherEntry) {
            throw new Error(
                "Dispatcher node disappeared from snapshot. "
                + "Did the graph change mid-submission?"
            );
        }

        // Single-node API output. We pass the FULL workflow as the
        // GUI-format payload so the run history still shows the user's
        // graph, but only the dispatcher executes locally.
        const singleOutput = { [dispatcherId]: dispatcherEntry };

        // Submit. Try app.queuePrompt for legacy compatibility, then
        // fall back to api.queuePrompt (post-0.20 surface).
        const submit = async () => {
            if (typeof app.queuePrompt === "function") {
                return app.queuePrompt(0, {
                    output: singleOutput,
                    workflow: fresh.workflow,
                });
            }
            if (api && typeof api.queuePrompt === "function") {
                return api.queuePrompt(0, {
                    output: singleOutput,
                    workflow: fresh.workflow,
                });
            }
            throw new Error(
                "No queuePrompt method found on app or api. "
                + "ComfyUI front-end version may be incompatible."
            );
        };

        await submit();
        appendStatusLine(node, "Dispatcher queued.");
    } catch (e) {
        console.error("rs-nodes runOnRunPod failed:", e);
        appendStatusLine(node, `ERROR: ${e?.message || e}`);
    }
}

// ---------------------------------------------------------------------------
// Event wiring (one-time, page-level)
// ---------------------------------------------------------------------------

function findNodeById(rawId) {
    if (rawId === undefined || rawId === null) return null;
    return app.graph?.getNodeById(Number(rawId)) || null;
}

api.addEventListener("rs.runpod.phase", (event) => {
    const d = event.detail || {};
    const node = findNodeById(d.node_id);
    if (!node) return;
    appendStatusLine(node, d.line || `[${d.step}/${d.total}] ${d.text || ""}`);
});

api.addEventListener("rs.runpod.log", (event) => {
    const d = event.detail || {};
    const node = findNodeById(d.node_id);
    if (!node) return;
    appendStatusLine(node, d.text || "");
});

api.addEventListener("rs.runpod.progress", (event) => {
    const d = event.detail || {};
    const node = findNodeById(d.node_id);
    if (!node) return;
    const label = d.node ? `node ${d.node}` : "";
    setProgressLine(node, d.value || 0, d.max || 1, label);
});

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "rs-nodes.RunPodDispatch",

    nodeCreated(node) {
        if (node.comfyClass !== "RSRunOnRunPod") return;

        // 1. Hide the JSON ferry widget.
        hideWorkflowWidget(node);

        // 2. Add the action button.
        node.addWidget(
            "button",
            "Run on RunPod",
            null,
            () => runOnRunPod(node),
        );

        // 3. Make sure the status panel exists.
        ensureStatusWidget(node);

        node.setSize(node.computeSize());
    },
});
