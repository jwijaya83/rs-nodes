import { app } from "../../scripts/app.js";

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

app.registerExtension({
    name: "rs-nodes.LTXVTrain",

    nodeCreated(node) {
        if (node.comfyClass === "RSLTXVPrepareDataset") {
            addSections(node, PREPARE_SECTIONS);
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
        }
    },
});
