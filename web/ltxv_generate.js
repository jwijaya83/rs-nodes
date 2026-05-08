import { app } from "../../scripts/app.js";

// Section definitions: [label, first_widget_name]
const GENERATE_SECTIONS = [
    ["Generation", "width"],
    ["Frame Injection", "first_strength"],
    ["Audio", "audio_cfg"],
    ["Guidance", "stg_scale"],
    ["Efficiency", "attention_mode"],
    ["Upscale", "upscale"],
    ["Output", "decode"],
    ["Scheduler", "max_shift"],
];

const EXTEND_SECTIONS = [
    ["Extension", "num_new_frames"],
    ["Frame Injection", "last_strength"],
    ["Efficiency", "attention_mode"],
    ["Upscale", "upscale"],
    ["Output", "decode"],
    ["Scheduler", "max_shift"],
];

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

        // Move header before the target widget
        const hdrIdx = node.widgets.indexOf(header);
        node.widgets.splice(hdrIdx, 1);
        const targetIdx = node.widgets.indexOf(target);
        node.widgets.splice(targetIdx, 0, header);
    }
}

function hookSeedWriteback(node) {
    // Backend returns {"ui": {"noise_seed": [seed]}} after generation.
    // Mirror the value back into the widget as a safety net (in case the
    // backend ever resolves a seed itself); also keeps the displayed value
    // truthful when the widget was set by preRollSeed below.
    const origOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        origOnExecuted?.apply(this, arguments);
        const v = message?.noise_seed;
        if (v && v.length) {
            const seedWidget = this.widgets?.find((w) => w.name === "noise_seed");
            if (seedWidget) {
                const num = Number(v[0]);
                if (Number.isFinite(num) && seedWidget.value !== num) {
                    seedWidget.value = num;
                    if (typeof seedWidget.callback === "function") {
                        try { seedWidget.callback(num); } catch {}
                    }
                    this.setDirtyCanvas?.(true, true);
                }
            }
        }
    };
}

function preRollSeed(node) {
    // Resolve seed_mode in the frontend BEFORE prompt submission so the
    // rolled noise_seed lands in the serialized workflow JSON. That way the
    // seed embedded in the output file's metadata matches the seed actually
    // used -- so dragging the file back in restores the original seed and
    // re-running with seed_mode='fixed' reproduces the exact output.
    const seedWidget = node.widgets?.find((w) => w.name === "noise_seed");
    const modeWidget = node.widgets?.find((w) => w.name === "seed_mode");
    if (!seedWidget || !modeWidget) return;
    const mode = modeWidget.value;
    let next = seedWidget.value;
    if (mode === "random") {
        // 53-bit safe-int range. Backend max is uint64 but JS Number can't
        // represent that without precision loss; 53 bits is plenty of entropy.
        next = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    } else if (mode === "increment") {
        next = (Number(seedWidget.value) + 1) % Number.MAX_SAFE_INTEGER;
    } else if (mode === "decrement") {
        next = Number(seedWidget.value) - 1;
        if (next < 0) next = Number.MAX_SAFE_INTEGER - 1;
    }
    if (next !== seedWidget.value) {
        seedWidget.value = next;
        if (typeof seedWidget.callback === "function") {
            try { seedWidget.callback(next); } catch {}
        }
    }
}

// Wrap whichever queue-submission method is actually wired up so every
// queue submission gets a fresh seed roll for any RSLTXVGenerate node.
// Bound + closure-saved original so chained extensions still work.
//
// The wrap MUST happen inside setup() rather than at module top-level —
// at module load time `app.queuePrompt` and `app.api` may not be defined
// yet, and any wrapping there gets clobbered by later ComfyUI init.
// 0.20.x in particular routes submissions through `app.api.queuePrompt`
// rather than the legacy `app.queuePrompt`, so we hook both.
function installSeedRollHook() {
    const wrap = (target, method, label) => {
        if (!target || typeof target[method] !== "function") return false;
        // Prevent double-wrapping if this extension's setup runs twice
        if (target[method].__rsRollWrapped) return true;
        const orig = target[method].bind(target);
        const wrapped = async function (...args) {
            try {
                for (const node of (app.graph?._nodes || [])) {
                    if (node.comfyClass === "RSLTXVGenerate") {
                        preRollSeed(node);
                    }
                }
            } catch (e) {
                console.warn(`rs-nodes preRollSeed failed (${label}):`, e);
            }
            return orig(...args);
        };
        wrapped.__rsRollWrapped = true;
        target[method] = wrapped;
        return true;
    };
    const w1 = wrap(app, "queuePrompt", "app.queuePrompt");
    const w2 = wrap(app.api, "queuePrompt", "app.api.queuePrompt");
    if (!w1 && !w2) {
        console.warn(
            "rs-nodes: could not hook any queue submission method; "
            + "seed_mode=random/increment/decrement will not roll seeds. "
            + "(Neither app.queuePrompt nor app.api.queuePrompt exists.)"
        );
    }
}

app.registerExtension({
    name: "rs-nodes.LTXVGenerate",

    setup() {
        installSeedRollHook();
    },

    nodeCreated(node) {
        if (node.comfyClass === "RSLTXVGenerate") {
            addSections(node, GENERATE_SECTIONS);
            hookSeedWriteback(node);
        } else if (node.comfyClass === "RSLTXVExtend") {
            addSections(node, EXTEND_SECTIONS);
        }
    },
});
