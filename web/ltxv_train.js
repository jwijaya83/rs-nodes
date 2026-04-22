import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PREPARE_SECTIONS = [
    ["Dataset", "media_folder"],
    ["Model Paths", "model_path"],
    ["Captioning", "caption_mode"],
    ["Options", "resolution_buckets"],
    ["Face Detection", "face_detection"],
    ["IC-LoRA", "conditioning_folder"],
];

const TRAIN_SECTIONS = [
    ["Model", "model"],
    ["Preset", "preset"],
    ["LoRA Config", "lora_rank"],
    ["Module Selection", "video_self_attention"],
    ["Training", "learning_rate"],
    ["Quantization", "quantization"],
    ["Strategy", "strategy"],
    ["Validation", "clip"],
    ["Checkpoints", "checkpoint_interval"],
    ["Resume", "resume_checkpoint"],
];

// Preset definitions: { modules: {toggle: bool}, rank, alpha }
const PRESETS = {
    "subject": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: false,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "style": {
        video_self_attention: true,
        video_cross_attention: false,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "motion": {
        video_self_attention: true,
        video_cross_attention: false,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "subject + style": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 32,
        lora_alpha: 32,
    },
    "all video": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 32,
        lora_alpha: 32,
    },
    "audio + video": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: true,
        audio_cross_attention: true,
        audio_feed_forward: true,
        video_attends_to_audio: true,
        audio_attends_to_video: true,
        lora_rank: 32,
        lora_alpha: 32,
    },
};

function applyPreset(node, presetName) {
    const preset = PRESETS[presetName];
    if (!preset) return; // "custom" — don't touch anything

    for (const [key, value] of Object.entries(preset)) {
        const widget = node.widgets.find((w) => w.name === key);
        if (widget) {
            widget.value = value;
        }
    }
}

function createSectionHeader(label) {
    const bar = document.createElement("div");
    bar.style.cssText =
        "width:100%;display:flex;align-items:center;gap:6px;" +
        "padding:6px 4px 2px;box-sizing:border-box;";

    const line1 = document.createElement("div");
    line1.style.cssText = "flex:0 0 8px;height:1px;background:#666;";

    const text = document.createElement("span");
    text.textContent = label;
    text.style.cssText =
        "color:#aaa;font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:0.5px;white-space:nowrap;";

    const line2 = document.createElement("div");
    line2.style.cssText = "flex:1;height:1px;background:#666;";

    bar.appendChild(line1);
    bar.appendChild(text);
    bar.appendChild(line2);
    return bar;
}

function addSections(node, sections) {
    for (const [label, firstWidget] of sections) {
        const target = node.widgets.find((w) => w.name === firstWidget);
        if (!target) continue;

        const header = node.addDOMWidget(
            "section_" + label.toLowerCase().replace(/\s+/g, "_"),
            "custom",
            createSectionHeader(label),
            { serialize: false }
        );

        const hdrIdx = node.widgets.indexOf(header);
        node.widgets.splice(hdrIdx, 1);
        const targetIdx = node.widgets.indexOf(target);
        node.widgets.splice(targetIdx, 0, header);
    }
}

// ---------------------------------------------------------------------------
// Training monitor link
// ---------------------------------------------------------------------------

// Training monitor is now a standalone page: training_monitor.html
// Open it in a separate browser tab for full-size, resizable charts.

// Attach a canvas-drawn stats panel to a prepare-dataset node. The backend
// pushes live counts via the "rs.prepper.status" event; we just set the
// widget value and repaint. Not an input field — purely a readout.
function ensurePrepperStatusWidget(node) {
    let w = node.widgets && node.widgets.find((w) => w.name === "status_display");
    if (w) return w;
    const LINE_H = 14;
    const PAD = 6;
    w = {
        name: "status_display",
        type: "custom",
        value: "(waiting for run...)",
        serialize: false,
        options: { serialize: false },
        draw(ctx, graphNode, width, posY, _slotH) {
            const lines = (this.value || "").split("\n");
            const boxH = Math.max(LINE_H + PAD * 2, lines.length * LINE_H + PAD * 2);
            const x = PAD;
            const w_ = width - PAD * 2;
            ctx.fillStyle = "#1a1a1a";
            ctx.strokeStyle = "#3a3a3a";
            ctx.lineWidth = 1;
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(x, posY, w_, boxH, 4);
            } else {
                ctx.rect(x, posY, w_, boxH);
            }
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "#d8d8d8";
            ctx.font = "12px ui-monospace, 'Cascadia Code', 'Consolas', monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            let y = posY + PAD;
            for (const line of lines) {
                ctx.fillText(line, x + PAD, y);
                y += LINE_H;
            }
            this.computedHeight = boxH;
        },
        computeSize(width) {
            const lines = (this.value || "").split("\n");
            return [width, Math.max(LINE_H + PAD * 2, lines.length * LINE_H + PAD * 2)];
        },
    };
    node.addCustomWidget(w);
    return w;
}

// The stats panel is only relevant when the user has defined a character
// reference folder — otherwise there are no per-character counts to show.
function charRefsFolderValue(node) {
    const w = node.widgets && node.widgets.find((w) => w.name === "character_refs_folder");
    return w && w.value ? String(w.value).trim() : "";
}

function removePrepperStatusWidget(node) {
    if (!node.widgets) return;
    const idx = node.widgets.findIndex((w) => w.name === "status_display");
    if (idx >= 0) {
        node.widgets.splice(idx, 1);
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
    }
}

function syncPrepperStatusVisibility(node) {
    const hasRefs = charRefsFolderValue(node).length > 0;
    if (hasRefs) {
        ensurePrepperStatusWidget(node);
    } else {
        removePrepperStatusWidget(node);
    }
}

// Listen once per page load for status updates.
api.addEventListener("rs.prepper.status", (event) => {
    const detail = event.detail || {};
    const { node_id, text } = detail;
    if (node_id === undefined || node_id === null) return;
    const target = app.graph.getNodeById(Number(node_id));
    if (!target) return;
    // Only show stats if the user has a character folder configured.
    if (charRefsFolderValue(target).length === 0) return;
    const widget = ensurePrepperStatusWidget(target);
    widget.value = text || "";
    target.setSize(target.computeSize());
    target.setDirtyCanvas(true, true);
});

app.registerExtension({
    name: "rs-nodes.LTXVTrain",

    nodeCreated(node) {
        if (node.comfyClass === "RSLTXVPrepareDataset") {
            addSections(node, PREPARE_SECTIONS);
            // Show / hide the live stats panel based on whether the user
            // has a character refs folder defined. Hook the widget so
            // toggling the folder value later adds or removes the panel.
            syncPrepperStatusVisibility(node);
            const refsWidget = node.widgets.find((w) => w.name === "character_refs_folder");
            if (refsWidget) {
                const orig = refsWidget.callback;
                refsWidget.callback = function (value) {
                    if (orig) orig.call(this, value);
                    syncPrepperStatusVisibility(node);
                };
            }
        } else if (node.comfyClass === "RSLTXVTrainLoRA") {
            addSections(node, TRAIN_SECTIONS);

            // Wire up preset dropdown to apply settings on change
            const presetWidget = node.widgets.find((w) => w.name === "preset");
            if (presetWidget) {
                const origCallback = presetWidget.callback;
                presetWidget.callback = function (value) {
                    if (origCallback) origCallback.call(this, value);
                    applyPreset(node, value);
                };
                // Apply initial preset
                applyPreset(node, presetWidget.value);
            }

            // Hide/show ROSE-specific settings based on optimizer selection
            const optimizerWidget = node.widgets.find((w) => w.name === "optimizer");
            const roseWidgets = ["rose_stabilize", "rose_weight_decay", "rose_wd_schedule"];
            function updateRoseVisibility() {
                const isRose = optimizerWidget && optimizerWidget.value === "rose";
                for (const name of roseWidgets) {
                    const w = node.widgets.find((w) => w.name === name);
                    if (w) w.type = isRose ? w._origType || "toggle" : "hidden";
                }
                node.setSize(node.computeSize());
            }
            if (optimizerWidget) {
                // Store original widget types
                for (const name of roseWidgets) {
                    const w = node.widgets.find((w) => w.name === name);
                    if (w) w._origType = w.type;
                }
                const origOptCallback = optimizerWidget.callback;
                optimizerWidget.callback = function (value) {
                    if (origOptCallback) origOptCallback.call(this, value);
                    updateRoseVisibility();
                };
                updateRoseVisibility();
            }

            // Add link to open training monitor in a new tab
            const monitorLink = document.createElement("div");
            monitorLink.style.cssText = "width:100%;padding:8px 4px;box-sizing:border-box;text-align:center;";
            monitorLink.innerHTML = '<a href="/extensions/rs-nodes/training_monitor.html" target="_blank" ' +
                'style="color:#4a9eff;text-decoration:none;font-size:12px;font-family:monospace;">' +
                'Open Training Monitor</a>';
            node.addDOMWidget("monitor_link", "custom", monitorLink, { serialize: false });
        }
    },
});
