import { app } from "../../scripts/app.js";

// Shared seed-pre-roll module for every rs-nodes generation node.
// Without this, seed_mode=random/increment/decrement runs entirely
// in the Python backend, so the seed only updates in the widget AFTER
// generation finishes — meaning the user can't see what seed is being
// used until the run is over. Pre-rolling in the frontend BEFORE the
// prompt is submitted means:
//   * the rolled seed lands in the serialized workflow JSON
//   * the widget shows the seed in use BEFORE the run starts
//   * the seed in the output file's metadata matches the seed actually
//     used (drag-back-in restores the right seed; seed_mode='fixed' on
//     the same value reproduces the exact output)

// Per-node config: which widget holds the seed and which holds the mode.
// LTXV uses 'noise_seed' (legacy from the KSampler convention); Flux2
// and Z-Image use the simpler 'seed'.
const SEED_CONFIGS = {
    "RSLTXVGenerate": { seedWidget: "noise_seed", modeWidget: "seed_mode" },
    "RSFlux2Generate": { seedWidget: "seed",       modeWidget: "seed_mode" },
    "RSZImageGenerate": { seedWidget: "seed",       modeWidget: "seed_mode" },
};

function preRollSeed(node) {
    const config = SEED_CONFIGS[node.comfyClass];
    if (!config) return;
    const seedWidget = node.widgets?.find((w) => w.name === config.seedWidget);
    const modeWidget = node.widgets?.find((w) => w.name === config.modeWidget);
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

function hookSeedWriteback(node) {
    // Backend returns {"ui": {<seedWidget>: [seed]}} after generation.
    // Mirror the value back to the widget as a safety net — useful if
    // the backend ever resolves a seed itself rather than honoring the
    // frontend pre-roll. Also keeps the displayed value consistent.
    const config = SEED_CONFIGS[node.comfyClass];
    if (!config) return;
    const origOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        origOnExecuted?.apply(this, arguments);
        const v = message?.[config.seedWidget];
        if (v && v.length) {
            const seedWidget = this.widgets?.find((w) => w.name === config.seedWidget);
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

// Wrap whichever queue-submission method is actually wired up so every
// submission gets a fresh seed roll on any of the supported nodes.
// Closure-saved original so chained extensions still work.
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
                    if (SEED_CONFIGS[node.comfyClass]) {
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
    name: "rs-nodes.SeedRoll",

    setup() {
        installSeedRollHook();
    },

    nodeCreated(node) {
        if (SEED_CONFIGS[node.comfyClass]) {
            hookSeedWriteback(node);
        }
    },
});
