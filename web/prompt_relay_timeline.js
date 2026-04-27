// RSPromptRelayTimeline — interactive canvas timeline.
//
// User-facing UX: type a Style line, click "+ Add", type each segment's prompt,
// drag blocks on a single timeline track. JSON is the output, hidden from the UI.

import { app } from "../../scripts/app.js";

// ----- Constants ---------------------------------------------------------
const PADDING_X     = 12;
const AXIS_Y        = 18;        // axis line y; labels render below this
const AXIS_AREA_H   = 40;        // axis area above the segment track
const TRACK_H       = 38;        // segment track height (proportional, not dominating)
const CANVAS_H      = AXIS_AREA_H + TRACK_H + 8;
const EDGE_HOT_PX   = 9;
const HANDLE_PX     = 4;
const ADJACENT_EPS  = 1e-3;     // segments are "touching" if their boundaries are within this many seconds
const SEG_RADIUS    = 4;
const BG            = "#1a1d22";
const SEG_FILL      = "rgba(120, 180, 255, 0.45)";
const SEG_FILL_SEL  = "rgba(120, 200, 255, 0.85)";
const SEG_STROKE    = "rgba(120, 180, 255, 1.0)";
const SEG_STROKE_SEL= "rgba(255, 255, 255, 1.0)";
const AXIS_COLOR    = "#888";
const TICK_COLOR    = "#555";
const TEXT_COLOR    = "#ddd";
const TEXT_DIM      = "#888";
const HINT_COLOR    = "#666";
const MIN_SEG_SEC   = 0.04;

// ----- State helpers -----------------------------------------------------
function getDataWidget(node) {
    return node.widgets?.find((w) => w.name === "timeline_data");
}
function getNumberWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function readState(node) {
    const w = getDataWidget(node);
    let raw = w?.value ?? "";
    try {
        const obj = JSON.parse(raw || "{}");
        return {
            global: typeof obj.global === "string" ? obj.global : "",
            segments: (Array.isArray(obj.segments) ? obj.segments : [])
                .filter(
                    (s) => s && typeof s === "object" && typeof s.prompt === "string"
                        && Number.isFinite(s.t_start) && Number.isFinite(s.t_end)
                        && s.t_end > s.t_start
                )
                .map((s) => ({
                    t_start: Number(s.t_start),
                    t_end: Number(s.t_end),
                    prompt: String(s.prompt),
                })),
        };
    } catch {
        return { global: "", segments: [] };
    }
}

function writeState(node, state) {
    const w = getDataWidget(node);
    if (!w) return;
    const segs = [...state.segments].sort((a, b) => a.t_start - b.t_start);
    const out = {
        global: state.global || "",
        segments: segs.map((s) => ({
            t_start: Math.round(s.t_start * 1000) / 1000,
            t_end:   Math.round(s.t_end   * 1000) / 1000,
            prompt:  s.prompt,
        })),
    };
    const json = JSON.stringify(out, null, 2);
    if (w.value !== json) {
        w.value = json;
        if (typeof w.callback === "function") {
            try { w.callback(w.value); } catch {}
        }
    }
}

function snapToFrame(t, fps, enabled) {
    if (!enabled || !Number.isFinite(fps) || fps <= 0) return t;
    return Math.round(t * fps) / fps;
}

// Re-tile segments to fill exactly [0, totalSec], preserving each segment's
// relative width. Mutates segments in place.
function rescaleToFill(segments, totalSec) {
    if (!segments.length || totalSec <= 0) return;
    const sorted = [...segments].sort((a, b) => a.t_start - b.t_start);
    const totalSpan = sorted.reduce((s, x) => s + Math.max(x.t_end - x.t_start, 0), 0);
    let cursor = 0;
    if (totalSpan <= 0) {
        const w = totalSec / sorted.length;
        for (const s of sorted) {
            s.t_start = cursor;
            cursor += w;
            s.t_end = cursor;
        }
    } else {
        const scale = totalSec / totalSpan;
        for (const s of sorted) {
            const w = Math.max((s.t_end - s.t_start) * scale, 0);
            s.t_start = cursor;
            cursor += w;
            s.t_end = cursor;
        }
    }
    // Pin last edge to totalSec to absorb floating-point drift.
    sorted[sorted.length - 1].t_end = totalSec;
}

// ----- Geometry ----------------------------------------------------------
function timeToX(t, totalSec, cssW) {
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    return PADDING_X + (t / Math.max(totalSec, 1e-6)) * innerW;
}
function xToTime(x, totalSec, cssW) {
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    return ((x - PADDING_X) / innerW) * Math.max(totalSec, 1e-6);
}

function hitTest(state, totalSec, cssW, mx, my) {
    if (my < AXIS_AREA_H || my > AXIS_AREA_H + TRACK_H) return null;
    // First pass: prefer edge hits (resize) over body hits (move) — even if a body
    // overlaps another segment's edge, the edge wins for grab-the-trim feel.
    let bodyHit = null;
    const ordered = state.segments
        .map((s, i) => ({ s, i }))
        .sort((a, b) => a.s.t_start - b.s.t_start);
    for (const { s, i } of ordered) {
        const x0 = timeToX(s.t_start, totalSec, cssW);
        const x1 = timeToX(s.t_end, totalSec, cssW);
        if (mx >= x0 - EDGE_HOT_PX && mx <= x0 + EDGE_HOT_PX) return { index: i, region: "left" };
        if (mx >= x1 - EDGE_HOT_PX && mx <= x1 + EDGE_HOT_PX) return { index: i, region: "right" };
        if (mx > x0 + EDGE_HOT_PX && mx < x1 - EDGE_HOT_PX) bodyHit = { index: i, region: "body" };
    }
    return bodyHit;
}

// Find rolling-trim neighbour: returns the adjacent segment whose boundary
// touches `seg`'s edge on the given side, or null if there's a gap.
function findAdjacent(state, seg, side) {
    const sorted = [...state.segments].sort((a, b) => a.t_start - b.t_start);
    const idx = sorted.indexOf(seg);
    if (idx < 0) return null;
    if (side === "right") {
        const next = sorted[idx + 1];
        if (next && Math.abs(next.t_start - seg.t_end) <= ADJACENT_EPS) return next;
    } else if (side === "left") {
        const prev = sorted[idx - 1];
        if (prev && Math.abs(prev.t_end - seg.t_start) <= ADJACENT_EPS) return prev;
    }
    return null;
}


// ----- Drawing -----------------------------------------------------------
function draw(canvas, node, runtime) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
    const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);

    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(CANVAS_H * dpr)) {
        canvas.width = Math.floor(cssW * dpr);
        canvas.height = Math.floor(CANVAS_H * dpr);
        canvas.style.height = CANVAS_H + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssW, CANVAS_H);

    // Pick a tick stride that keeps labels readable (>= ~50px apart).
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    const pxPerSec = innerW / Math.max(totalSec, 1e-6);
    const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60];
    let majorStep = candidates[candidates.length - 1];
    for (const c of candidates) {
        if (c * pxPerSec >= 50) { majorStep = c; break; }
    }
    const minorStep = majorStep >= 1 ? majorStep / (majorStep >= 5 ? 5 : 2) : majorStep / 2;
    const frameSec = 1 / Math.max(fps, 0.01);

    // Minor ticks: minorStep, plus per-frame faint ticks if dense enough.
    ctx.strokeStyle = TICK_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 0; t <= totalSec + 1e-6; t += minorStep) {
        const x = timeToX(t, totalSec, cssW);
        ctx.moveTo(x + 0.5, AXIS_Y);
        ctx.lineTo(x + 0.5, AXIS_Y + 5);
    }
    ctx.stroke();
    if (frameSec * pxPerSec >= 4) {
        ctx.strokeStyle = "#3a3d44";
        ctx.beginPath();
        for (let t = 0; t <= totalSec + 1e-6; t += frameSec) {
            const x = timeToX(t, totalSec, cssW);
            ctx.moveTo(x + 0.5, AXIS_Y);
            ctx.lineTo(x + 0.5, AXIS_Y + 3);
        }
        ctx.stroke();
    }

    // Axis line
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING_X, AXIS_Y + 0.5);
    ctx.lineTo(cssW - PADDING_X, AXIS_Y + 0.5);
    ctx.stroke();

    // Major ticks + labels
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = "11px sans-serif";
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    ctx.strokeStyle = AXIS_COLOR;
    ctx.beginPath();
    const dec = majorStep >= 1 ? 0 : (majorStep >= 0.1 ? 1 : 2);
    for (let t = 0; t <= totalSec + 1e-6; t += majorStep) {
        const x = timeToX(t, totalSec, cssW);
        ctx.moveTo(x + 0.5, AXIS_Y);
        ctx.lineTo(x + 0.5, AXIS_Y + 9);
        const label = `${t.toFixed(dec)}s`;
        ctx.fillText(label, x, AXIS_Y + 11);
    }
    ctx.stroke();

    // Track baseline
    const trackY = AXIS_AREA_H;
    ctx.strokeStyle = "#2a2d33";
    ctx.lineWidth = 1;
    ctx.strokeRect(PADDING_X + 0.5, trackY + 0.5, innerW - 1, TRACK_H - 1);

    // Audio waveform — only the trimmed [in, out] portion mapped onto [0, totalSec].
    const audio = runtime.audio;
    if (audio && audio.buffer) {
        const targetW = Math.floor(innerW);
        if (!audio.peaks || audio.peaksW !== targetW) {
            audio.peaks = computePeaks(audio.buffer, targetW);
            audio.peaksW = targetW;
        }
        const inSec = runtime.trimIn();
        const outSec = runtime.trimOut();
        const trimDur = Math.max(outSec - inSec, 1e-6);
        const peaksLen = audio.peaks.length;
        // Source range in peaks index space corresponding to [inSec, outSec]:
        const srcStart = (inSec / Math.max(audio.duration, 1e-6)) * peaksLen;
        const srcEnd   = (outSec / Math.max(audio.duration, 1e-6)) * peaksLen;
        ctx.fillStyle = "rgba(140, 200, 240, 0.30)";
        const cy = trackY + TRACK_H / 2;
        const halfH = (TRACK_H - 6) / 2;
        // Iterate destination pixels across [0, totalSec] mapped to [srcStart, srcEnd].
        const visibleEndX = timeToX(Math.min(trimDur, totalSec), totalSec, cssW);
        const visiblePxW = Math.max(visibleEndX - PADDING_X, 1);
        for (let px = 0; px <= visiblePxW; px++) {
            const u = px / visiblePxW;        // 0..1 across visible portion
            const peakIdx = Math.floor(srcStart + u * (srcEnd - srcStart));
            if (peakIdx < 0 || peakIdx >= peaksLen) continue;
            const v = audio.peaks[peakIdx];
            const h = Math.max(v * halfH, 0.5);
            const x = PADDING_X + px;
            ctx.fillRect(x, cy - h, 1, h * 2);
        }
    }

    // Segments — all on the SAME track row, time-ordered. Overlaps are
    // visualised by translucent fill stacking; selected drawn on top.
    const sel = runtime.getSelected();
    const renderOrder = runtime.state.segments
        .map((s, i) => ({ s, i }))
        .sort((a, b) => a.s.t_start - b.s.t_start)
        .filter(({ s }) => s !== sel);
    if (sel && runtime.state.segments.indexOf(sel) >= 0) {
        renderOrder.push({ s: sel, i: runtime.state.segments.indexOf(sel) });
    }

    const audioLoaded = !!(runtime.audio && runtime.audio.buffer);
    // When audio is loaded, drop fill alpha so the waveform shows through. The
    // selection cue switches to a strong bright border around the clip rather
    // than a heavier fill — keeps the waveform readable inside selected clips.
    const fillUnsel  = audioLoaded ? "rgba(120, 180, 255, 0.16)" : SEG_FILL;
    const fillSel    = audioLoaded ? "rgba(160, 220, 255, 0.22)" : SEG_FILL_SEL;
    const strokeUnsel = audioLoaded ? "rgba(120, 180, 255, 0.95)" : SEG_STROKE;
    const strokeSel   = audioLoaded ? "rgba(255, 235, 100, 1.0)"   : SEG_STROKE_SEL;

    for (const { s, i } of renderOrder) {
        const isSel = s === sel;
        const x0 = timeToX(s.t_start, totalSec, cssW);
        const x1 = timeToX(s.t_end, totalSec, cssW);
        const w = Math.max(x1 - x0, 2);

        ctx.fillStyle = isSel ? fillSel : fillUnsel;
        roundRect(ctx, x0, trackY + 2, w, TRACK_H - 4, SEG_RADIUS);
        ctx.fill();
        ctx.strokeStyle = isSel ? strokeSel : strokeUnsel;
        ctx.lineWidth = isSel ? (audioLoaded ? 2.5 : 2) : 1;
        roundRect(ctx, x0 + 0.5, trackY + 2.5, w - 1, TRACK_H - 5, SEG_RADIUS);
        ctx.stroke();
        ctx.lineWidth = 1;

        // Trim handles — always visible. When selected with audio loaded, make
        // them yellow to match the selection accent and keep the body see-through.
        const handleColor = isSel
            ? (audioLoaded ? "rgba(255, 235, 100, 1.0)" : "rgba(255,255,255,0.95)")
            : "rgba(255,255,255,0.55)";
        ctx.fillStyle = handleColor;
        ctx.fillRect(x0, trackY + 2, HANDLE_PX, TRACK_H - 4);
        ctx.fillRect(x1 - HANDLE_PX, trackY + 2, HANDLE_PX, TRACK_H - 4);

        // Label clipped to body
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + 4, trackY, Math.max(w - 8, 0), TRACK_H);
        ctx.clip();
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = "11px sans-serif";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        const label = s.prompt || `(segment ${i + 1})`;
        ctx.fillText(label, x0 + 6, trackY + TRACK_H / 2 - 4);
        ctx.fillStyle = TEXT_DIM;
        ctx.font = "9px sans-serif";
        ctx.fillText(`${s.t_start.toFixed(2)}–${s.t_end.toFixed(2)}s`, x0 + 6, trackY + TRACK_H - 6);
        ctx.restore();
    }

    if (runtime.state.segments.length === 0 && !(runtime.audio && runtime.audio.buffer)) {
        ctx.fillStyle = HINT_COLOR;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Click “+ Add” to start • drag body = move • drag edges = resize",
                     cssW / 2, trackY + TRACK_H / 2);
    }

    // Blade tool — the line IS the cursor. Spans the full canvas height so the
    // user can see where they are even if the mouse is in the axis area.
    if (runtime.tool === "blade" && runtime.bladeHoverX != null) {
        const bx = runtime.bladeHoverX;
        ctx.strokeStyle = "rgba(255, 80, 80, 0.95)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(bx + 0.5, 0);
        ctx.lineTo(bx + 0.5, CANVAS_H);
        ctx.stroke();
        ctx.setLineDash([]);
        // Tiny "blade tip" caret at the bottom for clarity.
        ctx.fillStyle = "rgba(255, 80, 80, 0.95)";
        ctx.beginPath();
        ctx.moveTo(bx - 4, CANVAS_H - 1);
        ctx.lineTo(bx + 4, CANVAS_H - 1);
        ctx.lineTo(bx,     CANVAS_H - 7);
        ctx.closePath();
        ctx.fill();
    }

    // Playhead — converted from audio-time to timeline-time (subtract trim in).
    if (runtime.audio && runtime.audio.buffer) {
        const inSec = runtime.trimIn();
        const tTimeline = Math.max(0, Math.min(runtime.audio.playheadSec - inSec, totalSec));
        const px = timeToX(tTimeline, totalSec, cssW);
        ctx.strokeStyle = "rgba(255, 200, 80, 0.95)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px + 0.5, AXIS_Y - 4);
        ctx.lineTo(px + 0.5, trackY + TRACK_H);
        ctx.stroke();
        ctx.fillStyle = "rgba(255, 200, 80, 0.95)";
        ctx.beginPath();
        ctx.moveTo(px - 4, AXIS_Y - 6);
        ctx.lineTo(px + 4, AXIS_Y - 6);
        ctx.lineTo(px,     AXIS_Y);
        ctx.closePath();
        ctx.fill();
    }
}

// Pure helper at module scope so draw() can see it.
function computePeaks(buffer, targetW) {
    // Two-channel-merged absolute peaks per pixel column.
    const ch = buffer.numberOfChannels;
    const data = buffer.getChannelData(0);
    const data2 = ch > 1 ? buffer.getChannelData(1) : null;
    const N = data.length;
    const cols = Math.max(targetW, 1);
    const out = new Float32Array(cols);
    const samplesPerCol = N / cols;
    for (let c = 0; c < cols; c++) {
        const s = Math.floor(c * samplesPerCol);
        const e = Math.min(N, Math.floor((c + 1) * samplesPerCol));
        let mx = 0;
        for (let i = s; i < e; i++) {
            const v1 = Math.abs(data[i]);
            const v2 = data2 ? Math.abs(data2[i]) : 0;
            const v = Math.max(v1, v2);
            if (v > mx) mx = v;
        }
        out[c] = mx;
    }
    return out;
}

function roundRect(ctx, x, y, w, h, r) {
    const rad = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
}

// ----- Widget wiring -----------------------------------------------------
function hideWidget(w) {
    if (!w) return;
    w.computeSize = () => [0, -4];
    try { Object.defineProperty(w, "type", { value: "hidden", configurable: true, writable: true }); }
    catch { w.type = "hidden"; }
    w.draw = () => {};
    if (w.element) {
        w.element.style.display = "none";
        if (w.element.parentElement) w.element.parentElement.style.display = "none";
    }
    if (w.inputEl) {
        w.inputEl.style.display = "none";
        if (w.inputEl.parentElement) w.inputEl.parentElement.style.display = "none";
    }
}

function hideTimelineDataWidget(node) {
    hideWidget(getDataWidget(node));
    hideWidget(node.widgets?.find((w) => w.name === "audio_in_sec"));
    hideWidget(node.widgets?.find((w) => w.name === "audio_out_sec"));
}

function attachTimeline(node) {
    if (node._rsTimelineAttached) return;
    node._rsTimelineAttached = true;

    hideTimelineDataWidget(node);

    // Fixed pixel heights for each row so flex-shrink can't collapse textareas.
    const H_LABEL  = 14;
    const H_STYLE  = 50;
    const H_TOOL   = 28;
    const H_EDITOR = 64;
    const GAP      = 6;
    const PADV     = 8;        // top + bottom padding inside root
    const TOTAL_H  = PADV + H_LABEL + GAP + H_STYLE + GAP + H_TOOL + GAP + CANVAS_H + GAP + H_LABEL + GAP + H_EDITOR;

    const childCSS = "flex-shrink:0;flex-grow:0;";

    // Build DOM
    const root = document.createElement("div");
    root.style.cssText =
        "width:100%;display:flex;flex-direction:column;gap:" + GAP + "px;" +
        "padding:" + (PADV / 2) + "px 6px;box-sizing:border-box;font-family:sans-serif;";

    const styleLabel = document.createElement("div");
    styleLabel.textContent = "STYLE  (always-on context)";
    styleLabel.style.cssText = childCSS +
        "height:" + H_LABEL + "px;line-height:" + H_LABEL + "px;" +
        "color:#aaa;font-size:10px;font-weight:bold;letter-spacing:0.5px;";

    const styleArea = document.createElement("textarea");
    styleArea.placeholder = "e.g. cinematic 35mm, warm tungsten light, woman in red dress";
    styleArea.style.cssText = childCSS +
        "width:100%;height:" + H_STYLE + "px;box-sizing:border-box;background:#222;color:#eee;" +
        "border:1px solid #444;border-radius:3px;padding:4px 6px;font:11px sans-serif;resize:none;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = childCSS +
        "height:" + H_TOOL + "px;display:flex;gap:6px;align-items:center;";
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.textContent = "+ Add";
    addBtn.style.cssText =
        "background:#2c5;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.textContent = "× Delete";
    delBtn.style.cssText =
        "background:#623;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    delBtn.disabled = true;
    delBtn.style.opacity = "0.5";
    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.textContent = "▶";
    playBtn.title = "Play / pause (L=play  K=pause  J=stop)";
    playBtn.style.cssText =
        "background:#234;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;min-width:30px;";
    playBtn.disabled = true;
    playBtn.style.opacity = "0.5";
    const stopBtn = document.createElement("button");
    stopBtn.type = "button";
    stopBtn.textContent = "■";
    stopBtn.title = "Rewind to in-point";

    const uploadBtn = document.createElement("button");
    uploadBtn.type = "button";
    uploadBtn.textContent = "📁 Upload";
    uploadBtn.title = "Upload an audio file to the input/ folder";
    uploadBtn.style.cssText =
        "background:#345;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";

    const resetIOBtn = document.createElement("button");
    resetIOBtn.type = "button";
    resetIOBtn.textContent = "Reset I/O";
    resetIOBtn.title = "Reset trim in/out to the full audio clip";
    resetIOBtn.style.cssText =
        "background:#444;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    resetIOBtn.disabled = true;
    resetIOBtn.style.opacity = "0.5";

    const bladeBtn = document.createElement("button");
    bladeBtn.type = "button";
    bladeBtn.textContent = "✂ Cut";
    bladeBtn.title = "Blade tool — click a segment to split it (shortcut: B • Esc to exit)";
    bladeBtn.style.cssText =
        "background:#444;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    stopBtn.style.cssText =
        "background:#234;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;min-width:30px;";
    stopBtn.disabled = true;
    stopBtn.style.opacity = "0.5";
    const timeLabel = document.createElement("span");
    timeLabel.style.cssText = "color:#aaa;font:11px monospace;margin-left:auto;";
    timeLabel.textContent = "0.00 / 0.00s";
    toolbar.appendChild(addBtn);
    toolbar.appendChild(delBtn);
    toolbar.appendChild(bladeBtn);
    toolbar.appendChild(uploadBtn);
    toolbar.appendChild(playBtn);
    toolbar.appendChild(stopBtn);
    toolbar.appendChild(resetIOBtn);
    toolbar.appendChild(timeLabel);

    const canvas = document.createElement("canvas");
    canvas.tabIndex = 0;
    canvas.style.cssText = childCSS +
        "width:100%;height:" + CANVAS_H + "px;display:block;border-radius:4px;outline:none;";
    canvas.width = 600;
    canvas.height = CANVAS_H;

    const editorLabel = document.createElement("div");
    editorLabel.textContent = "SELECTED SEGMENT";
    editorLabel.style.cssText = childCSS +
        "height:" + H_LABEL + "px;line-height:" + H_LABEL + "px;" +
        "color:#aaa;font-size:10px;font-weight:bold;letter-spacing:0.5px;";

    const editorArea = document.createElement("textarea");
    editorArea.placeholder = "(select a segment to edit its prompt)";
    editorArea.disabled = true;
    editorArea.style.cssText = childCSS +
        "width:100%;height:" + H_EDITOR + "px;box-sizing:border-box;background:#222;color:#eee;" +
        "border:1px solid #444;border-radius:3px;padding:4px 6px;font:11px sans-serif;resize:none;opacity:0.5;";

    root.appendChild(styleLabel);
    root.appendChild(styleArea);
    root.appendChild(toolbar);
    root.appendChild(canvas);
    root.appendChild(editorLabel);
    root.appendChild(editorArea);

    const widget = node.addDOMWidget("rs_timeline_ui", "ui", root, { serialize: false });
    // Fixed total height — ComfyUI sums widget computeSize values to lay out the node.
    widget.computeSize = function (width) {
        return [width || 240, TOTAL_H];
    };

    // ----- Runtime state --------------------------------------------------
    const runtime = {
        state: readState(node),
        selectedRef: null,    // direct reference to a segment object (stable across edits)
        drag: null,
        tool: "select",       // "select" | "blade"
        bladeHoverX: null,    // when blade is active, cursor x for the preview line
        // Audio playback
        audio: {
            ctx: null,            // AudioContext
            buffer: null,         // AudioBuffer
            duration: 0,          // seconds
            peaks: null,          // Float32Array of waveform peaks (precomputed)
            peaksW: 0,            // pixel-width the peaks were computed at
            sourceNode: null,     // BufferSource currently playing
            startCtxTime: 0,      // ctx.currentTime when playback started
            startSec: 0,          // playback offset when started
            playheadSec: 0,       // current playhead in seconds
            playing: false,
            anim: null,           // requestAnimationFrame handle
            currentFile: null,    // last fetched filename
            scrubSource: null,    // short-chunk source for scrub audio
            lastScrubT: 0,        // ctx.currentTime of last scrub chunk start
        },
        getSelected() { return runtime.selectedRef; },
    };
    styleArea.value = runtime.state.global || "";

    // ----- Audio: fetch + decode -----------------------------------------
    function getOrCreateAudioCtx() {
        if (!runtime.audio.ctx) {
            const Ctor = window.AudioContext || window.webkitAudioContext;
            if (!Ctor) return null;
            runtime.audio.ctx = new Ctor();
        }
        return runtime.audio.ctx;
    }

    function viewURLs(filename) {
        if (!filename || filename === "(none)") return [];
        const params = new URLSearchParams({ filename, type: "input" });
        const qs = params.toString();
        // Try the standard endpoint first; some ComfyUI builds also expose /api/view.
        return [`/view?${qs}`, `/api/view?${qs}`];
    }

    async function loadAudio(filename) {
        const urls = viewURLs(filename);
        if (urls.length === 0) {
            stopPlayback();
            runtime.audio.buffer = null;
            runtime.audio.duration = 0;
            runtime.audio.peaks = null;
            runtime.audio.currentFile = null;
            runtime.audio.lastError = null;
            updateAudioUI();
            redraw();
            return;
        }
        if (runtime.audio.currentFile === filename && runtime.audio.buffer) return;
        const ctx = getOrCreateAudioCtx();
        if (!ctx) {
            runtime.audio.lastError = "AudioContext unavailable";
            updateAudioUI();
            return;
        }
        runtime.audio.lastError = "loading…";
        updateAudioUI();
        console.log("RSPromptRelayTimeline: loadAudio", filename);
        try {
            let resp = null;
            let lastErr = null;
            for (const url of urls) {
                try {
                    const r = await fetch(url);
                    if (r.ok) { resp = r; break; }
                    lastErr = `HTTP ${r.status} at ${url}`;
                } catch (e) {
                    lastErr = `${e} at ${url}`;
                }
            }
            if (!resp) throw new Error(lastErr || "all view URLs failed");
            const ab = await resp.arrayBuffer();
            console.log("RSPromptRelayTimeline: fetched", ab.byteLength, "bytes");
            const buf = await ctx.decodeAudioData(ab.slice(0));
            console.log("RSPromptRelayTimeline: decoded", buf.duration, "s @", buf.sampleRate, "Hz,", buf.numberOfChannels, "ch");
            stopPlayback();
            runtime.audio.buffer = buf;
            runtime.audio.duration = buf.duration;
            runtime.audio.peaks = null;       // recomputed per-canvas-width on draw
            runtime.audio.peaksW = 0;
            runtime.audio.currentFile = filename;
            // Reset trim to the full clip on a new audio load (unless the workflow
            // already has a saved trim within the clip's bounds).
            const wi = node.widgets?.find((x) => x.name === "audio_in_sec");
            const wo = node.widgets?.find((x) => x.name === "audio_out_sec");
            const savedIn  = wi ? Number(wi.value) || 0 : 0;
            const savedOut = wo ? Number(wo.value) || 0 : 0;
            const validSaved = savedOut > savedIn && savedOut <= buf.duration + 1e-3;
            if (!validSaved) {
                setTrim(0, buf.duration);
            } else {
                // Saved trim valid — just re-pin total_duration_sec to match.
                setTrim(savedIn, savedOut);
            }
            runtime.audio.playheadSec = trimInSec();
            runtime.audio.lastError = null;
            updateAudioUI();
            redraw();
        } catch (err) {
            console.error("RSPromptRelayTimeline: audio load failed:", err);
            runtime.audio.buffer = null;
            runtime.audio.duration = 0;
            runtime.audio.peaks = null;
            runtime.audio.currentFile = null;
            runtime.audio.lastError = String(err && err.message || err);
            updateAudioUI();
            redraw();
        }
    }

    // Play a short chunk for audible scrubbing/nudging. Throttled and
    // single-source-at-a-time so rapid moves don't queue up.
    function scrubAudio(fromSec, durSec = 0.10) {
        const a = runtime.audio;
        if (!a.buffer || !a.ctx) return;
        const now = a.ctx.currentTime;
        if (now - a.lastScrubT < 0.03) return; // ~30 Hz cap
        a.lastScrubT = now;
        if (a.ctx.state === "suspended") {
            try { a.ctx.resume(); } catch {}
        }
        if (a.scrubSource) {
            try { a.scrubSource.stop(0); } catch {}
            try { a.scrubSource.disconnect(); } catch {}
            a.scrubSource = null;
        }
        const offset = Math.max(0, Math.min(fromSec, a.duration - 0.001));
        const len = Math.max(0.02, Math.min(durSec, a.duration - offset));
        if (len <= 0) return;
        const src = a.ctx.createBufferSource();
        src.buffer = a.buffer;
        src.connect(a.ctx.destination);
        try { src.start(0, offset, len); } catch (e) { return; }
        a.scrubSource = src;
    }

    function getReverseBuffer() {
        const a = runtime.audio;
        if (!a.buffer || !a.ctx) return null;
        if (a.reverseBuffer && a.reverseBufferOf === a.buffer) return a.reverseBuffer;
        const src = a.buffer;
        const ch = src.numberOfChannels;
        const length = src.length;
        const rev = a.ctx.createBuffer(ch, length, src.sampleRate);
        for (let c = 0; c < ch; c++) {
            const ind = src.getChannelData(c);
            const out = rev.getChannelData(c);
            for (let i = 0; i < length; i++) out[i] = ind[length - 1 - i];
        }
        a.reverseBuffer = rev;
        a.reverseBufferOf = src;
        return rev;
    }

    function startPlayback(fromSec = null, dir = 1) {
        const a = runtime.audio;
        if (!a.buffer || !a.ctx) return;
        stopPlayback();
        const inSec = trimInSec();
        const outSec = trimOutSec();
        let offset = fromSec != null ? fromSec : a.playheadSec;
        offset = Math.max(inSec, Math.min(offset, outSec));

        let buf, srcOffset, playLen;
        if (dir < 0) {
            buf = getReverseBuffer();
            if (!buf) return;
            // Reverse buffer position 0 = original duration; original t maps to (audioDuration - t).
            srcOffset = Math.max(0, a.duration - offset);
            playLen = Math.max(0.001, offset - inSec);
            if (offset <= inSec + 1e-3) return; // already at the start, nothing to play
        } else {
            buf = a.buffer;
            srcOffset = offset;
            playLen = Math.max(0.001, outSec - offset);
            if (offset >= outSec - 1e-3) return; // already at the end
        }

        const src = a.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(a.ctx.destination);
        src.onended = () => {
            if (a.sourceNode === src) {
                a.sourceNode = null;
                a.playing = false;
                a.playDir = 0;
                a.playheadSec = dir < 0 ? trimInSec() : trimOutSec();
                cancelAnimationFrame(a.anim);
                a.anim = null;
                updateAudioUI();
                redraw();
            }
        };
        src.start(0, srcOffset, playLen);
        a.sourceNode = src;
        a.startCtxTime = a.ctx.currentTime;
        a.startSec = offset;
        a.playDir = dir < 0 ? -1 : 1;
        a.playing = true;
        const tick = () => {
            if (!a.playing) return;
            const elapsed = a.ctx.currentTime - a.startCtxTime;
            const t = a.startSec + a.playDir * elapsed;
            a.playheadSec = a.playDir < 0
                ? Math.max(inSec, t)
                : Math.min(outSec, t);
            redraw();
            a.anim = requestAnimationFrame(tick);
        };
        a.anim = requestAnimationFrame(tick);
        updateAudioUI();
    }

    // ----- Audio trim helpers (also exposed on runtime for draw()) -------
    runtime.trimIn  = () => trimInSec();
    runtime.trimOut = () => trimOutSec();

    function trimInSec() {
        const w = node.widgets?.find((x) => x.name === "audio_in_sec");
        return w ? Math.max(0, Number(w.value) || 0) : 0;
    }
    function trimOutSec() {
        const w = node.widgets?.find((x) => x.name === "audio_out_sec");
        const v = w ? Number(w.value) || 0 : 0;
        if (v <= 0 || v <= trimInSec()) return runtime.audio.duration || 0;
        return Math.min(v, runtime.audio.duration || v);
    }
    function setTrim(inSec, outSec) {
        const wi = node.widgets?.find((x) => x.name === "audio_in_sec");
        const wo = node.widgets?.find((x) => x.name === "audio_out_sec");
        if (wi) wi.value = Math.max(0, Math.round(inSec * 1000) / 1000);
        if (wo) wo.value = Math.max(0, Math.round(outSec * 1000) / 1000);
        // Sync total_duration_sec to the trim length so the timeline matches.
        const totalW = getNumberWidget(node, "total_duration_sec");
        if (totalW) {
            const newTotal = Math.max(0.1, Math.round((outSec - inSec) * 100) / 100);
            if (Math.abs(Number(totalW.value) - newTotal) > 1e-3) {
                totalW.value = newTotal;
                if (typeof totalW.callback === "function") {
                    try { totalW.callback(totalW.value); } catch {}
                }
            }
        }
    }

    function stopPlayback() {
        const a = runtime.audio;
        if (a.sourceNode) {
            try { a.sourceNode.onended = null; a.sourceNode.stop(0); } catch {}
            try { a.sourceNode.disconnect(); } catch {}
            a.sourceNode = null;
        }
        a.playing = false;
        if (a.anim) { cancelAnimationFrame(a.anim); a.anim = null; }
    }

    function updateAudioUI() {
        const a = runtime.audio;
        const enabled = !!a.buffer;
        playBtn.disabled = !enabled;
        stopBtn.disabled = !enabled;
        resetIOBtn.disabled = !enabled;
        playBtn.style.opacity = enabled ? "1" : "0.5";
        stopBtn.style.opacity = enabled ? "1" : "0.5";
        resetIOBtn.style.opacity = enabled ? "1" : "0.5";
        playBtn.textContent = a.playing ? "⏸" : "▶";
        if (!enabled) {
            timeLabel.textContent = a.lastError ? `audio: ${a.lastError}` : "no audio loaded";
            timeLabel.style.color = a.lastError && a.lastError !== "loading…" ? "#f88" : "#aaa";
            return;
        }
        const inT = trimInSec();
        const outT = trimOutSec();
        const dur = outT - inT;
        const cur = a.playheadSec.toFixed(2);
        timeLabel.style.color = "#aaa";
        timeLabel.textContent =
            `▸ ${cur}s   IN ${inT.toFixed(2)}  OUT ${outT.toFixed(2)}  DUR ${dur.toFixed(2)}s`;
    }

    playBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const a = runtime.audio;
        if (!a.buffer) return;
        if (a.playing) {
            stopPlayback();
            updateAudioUI();
            redraw();
        } else {
            // Resume audio context if suspended (browser autoplay policy).
            if (a.ctx && a.ctx.state === "suspended") {
                try { await a.ctx.resume(); } catch (err) { console.warn("ctx.resume failed:", err); }
            }
            // If playhead has reached the trim end, restart from the trim start.
            const trimEnd = trimOutSec();
            const from = a.playheadSec >= trimEnd - 0.001 ? trimInSec() : a.playheadSec;
            startPlayback(from);
        }
    });
    stopBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        stopPlayback();
        runtime.audio.playheadSec = trimInSec();
        updateAudioUI();
        redraw();
    });

    resetIOBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const a = runtime.audio;
        if (!a.buffer) return;
        stopPlayback();
        setTrim(0, a.duration);
        a.playheadSec = 0;
        updateAudioUI();
        redraw();
    });

    function setTool(tool) {
        runtime.tool = tool;
        if (tool === "blade") {
            bladeBtn.style.background = "#b85";
            canvas.style.cursor = "none";   // the rendered line replaces the OS cursor
        } else {
            bladeBtn.style.background = "#444";
            runtime.bladeHoverX = null;
            canvas.style.cursor = "default";
        }
        redraw();
    }

    bladeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        setTool(runtime.tool === "blade" ? "select" : "blade");
        canvas.focus();
    });

    uploadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = "audio/*,video/*";
        inp.style.display = "none";
        inp.addEventListener("change", async () => {
            const file = inp.files && inp.files[0];
            if (!file) return;
            runtime.audio.lastError = `uploading ${file.name}…`;
            updateAudioUI();
            const fd = new FormData();
            fd.append("image", file, file.name);  // /upload/image accepts any file
            fd.append("type", "input");
            fd.append("overwrite", "false");
            const endpoints = ["/api/upload/image", "/upload/image"];
            let savedName = null;
            let lastErr = null;
            for (const ep of endpoints) {
                try {
                    const resp = await fetch(ep, { method: "POST", body: fd });
                    if (!resp.ok) { lastErr = `HTTP ${resp.status} at ${ep}`; continue; }
                    const result = await resp.json();
                    savedName = result.name || result.filename || file.name;
                    if (result.subfolder) savedName = result.subfolder + "/" + savedName;
                    break;
                } catch (err) {
                    lastErr = `${err} at ${ep}`;
                }
            }
            if (!savedName) {
                runtime.audio.lastError = "upload failed: " + (lastErr || "unknown");
                updateAudioUI();
                return;
            }
            // Add to widget options if not already present, then select it.
            const w = node.widgets?.find((x) => x.name === "audio_file");
            if (w) {
                if (Array.isArray(w.options?.values) && !w.options.values.includes(savedName)) {
                    w.options.values.push(savedName);
                    w.options.values.sort();
                }
                w.value = savedName;
                if (typeof w.callback === "function") {
                    try { w.callback(savedName); } catch {}
                }
            }
            // Polling will also catch the value change and call loadAudio.
            runtime.audio.lastError = null;
            updateAudioUI();
        });
        document.body.appendChild(inp);
        inp.click();
        setTimeout(() => inp.remove(), 1000);
    });

    // Keep focus on the canvas so JKL/IO shortcuts work after toolbar clicks.
    [addBtn, delBtn, bladeBtn, playBtn, stopBtn, uploadBtn, resetIOBtn].forEach((b) => {
        b.addEventListener("click", () => { try { canvas.focus(); } catch {} });
    });

    function commit() {
        writeState(node, runtime.state);
        redraw();
    }
    function redraw() {
        const sel = runtime.getSelected();
        if (sel && runtime.state.segments.indexOf(sel) >= 0) {
            if (document.activeElement !== editorArea || editorArea.value !== sel.prompt) {
                editorArea.value = sel.prompt;
            }
            editorArea.disabled = false;
            editorArea.style.opacity = "1";
            delBtn.disabled = false;
            delBtn.style.opacity = "1";
        } else {
            runtime.selectedRef = null;
            editorArea.value = "";
            editorArea.disabled = true;
            editorArea.style.opacity = "0.5";
            delBtn.disabled = true;
            delBtn.style.opacity = "0.5";
        }
        draw(canvas, node, runtime);
    }

    // Resize observer just triggers a canvas redraw when the node width changes.
    const ro = new ResizeObserver(() => { draw(canvas, node, runtime); });
    ro.observe(root);

    // ----- Listeners ------------------------------------------------------
    styleArea.addEventListener("input", () => {
        runtime.state.global = styleArea.value;
        writeState(node, runtime.state);
    });

    editorArea.addEventListener("input", () => {
        const sel = runtime.getSelected();
        if (!sel) return;
        sel.prompt = editorArea.value;
        commit();
    });

    addBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);

        const N = runtime.state.segments.length;
        if (N === 0) {
            const seg = { prompt: "", t_start: 0, t_end: totalSec };
            runtime.state.segments.push(seg);
            runtime.selectedRef = seg;
            commit();
            editorArea.focus();
            return;
        }

        // New segment gets the equal-share default size. Existing segments
        // preserve their relative widths and rescale to fit the remaining space.
        const newLen = totalSec / (N + 1);
        rescaleToFill(runtime.state.segments, totalSec - newLen);
        const newSeg = { prompt: "", t_start: totalSec - newLen, t_end: totalSec };
        runtime.state.segments.push(newSeg);
        runtime.selectedRef = newSeg;
        commit();
        editorArea.focus();
    });

    delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const sel = runtime.getSelected();
        if (!sel) return;
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        runtime.state.segments = runtime.state.segments.filter((s) => s !== sel);
        runtime.selectedRef = null;
        rescaleToFill(runtime.state.segments, totalSec);
        commit();
    });

    function canvasMouse(e) {
        // Convert from VISUAL pixels (post-transform — workflow zoom etc.) to
        // LAYOUT pixels (the coordinate space the drawing code uses).
        const r = canvas.getBoundingClientRect();
        const sx = r.width  > 0 ? canvas.clientWidth  / r.width  : 1;
        const sy = r.height > 0 ? canvas.clientHeight / r.height : 1;
        return {
            x: (e.clientX - r.left) * sx,
            y: (e.clientY - r.top)  * sy,
        };
    }

    canvas.addEventListener("mousedown", (e) => {
        canvas.focus();
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);
        const cssW = canvas.clientWidth || 600;
        const { x, y } = canvasMouse(e);

        // Blade tool: click on a segment to split it.
        if (runtime.tool === "blade") {
            if (y >= AXIS_AREA_H && y <= AXIS_AREA_H + TRACK_H) {
                const tT = xToTime(x, totalSec, cssW);
                const seg = runtime.state.segments.find((s) => tT > s.t_start && tT < s.t_end);
                if (seg) {
                    let splitT = snapToFrame(tT, fps, !e.shiftKey);
                    splitT = Math.max(seg.t_start + MIN_SEG_SEC,
                                      Math.min(splitT, seg.t_end - MIN_SEG_SEC));
                    if (splitT > seg.t_start + MIN_SEG_SEC * 0.5 &&
                        splitT < seg.t_end - MIN_SEG_SEC * 0.5) {
                        const newSeg = {
                            prompt: seg.prompt,  // inherit prompt — user edits the second half
                            t_start: splitT,
                            t_end: seg.t_end,
                        };
                        seg.t_end = splitT;
                        runtime.state.segments.push(newSeg);
                        runtime.selectedRef = newSeg;
                        commit();
                    }
                }
            }
            e.preventDefault();
            return;
        }

        // Click in axis area = seek + start scrub drag (when audio is loaded).
        if (runtime.audio.buffer && y < AXIS_AREA_H) {
            // Resume context now (user gesture) so subsequent scrub chunks play.
            if (runtime.audio.ctx && runtime.audio.ctx.state === "suspended") {
                try { runtime.audio.ctx.resume(); } catch {}
            }
            const seekTo = (xx) => {
                const tT = Math.max(0, Math.min(xToTime(xx, totalSec, cssW), totalSec));
                const tA = trimInSec() + tT;
                runtime.audio.playheadSec = Math.max(trimInSec(), Math.min(tA, trimOutSec()));
            };
            const wasPlaying = runtime.audio.playing;
            if (wasPlaying) stopPlayback();
            seekTo(x);
            scrubAudio(runtime.audio.playheadSec, 0.12);
            runtime.drag = { kind: "scrub", resumeAfter: false };
            updateAudioUI();
            redraw();
            e.preventDefault();
            return;
        }
        const hit = hitTest(runtime.state, totalSec, cssW, x, y);
        if (hit) {
            const seg = runtime.state.segments[hit.index];
            runtime.selectedRef = seg;
            const t = xToTime(x, totalSec, cssW);
            let partner = null;
            // Snapshot all fixed pivots / widths up-front for the body-drag swap logic.
            const fixedPivots = new Map();
            const fixedWidths = new Map();
            for (const s of runtime.state.segments) {
                fixedPivots.set(s, (s.t_start + s.t_end) / 2);
                fixedWidths.set(s, s.t_end - s.t_start);
            }
            if (hit.region === "left") {
                partner = findAdjacent(runtime.state, seg, "left");
            } else if (hit.region === "right") {
                partner = findAdjacent(runtime.state, seg, "right");
            }
            runtime.drag = {
                kind: hit.region,
                seg,
                mouseT0: t,
                segT0: seg.t_start,
                segT1: seg.t_end,
                origLen: seg.t_end - seg.t_start,
                partner,
                partnerT0: partner ? partner.t_start : null,
                partnerT1: partner ? partner.t_end : null,
                fixedPivots,
                fixedWidths,
            };
            redraw();
            e.preventDefault();
        } else {
            runtime.selectedRef = null;
            redraw();
        }
    });

    function onMove(e) {
        if (!runtime.drag) {
            const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
            const { x, y } = canvasMouse(e);
            const cssW = canvas.clientWidth || 600;
            // Blade tool: track cursor x for the preview line; OS cursor is hidden.
            if (runtime.tool === "blade") {
                runtime.bladeHoverX = x;
                canvas.style.cursor = "none";
                redraw();
                return;
            }
            // Audio scrub cursor in axis area
            if (runtime.audio.buffer && y < AXIS_AREA_H) {
                canvas.style.cursor = "ew-resize";
                return;
            }
            const hit = hitTest(runtime.state, totalSec, cssW, x, y);
            canvas.style.cursor = !hit ? "default" :
                hit.region === "left" || hit.region === "right" ? "ew-resize" : "move";
            return;
        }
        // Audio scrub
        if (runtime.drag.kind === "scrub") {
            const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
            const cssW = canvas.clientWidth || 600;
            const { x } = canvasMouse(e);
            const tT = Math.max(0, Math.min(xToTime(x, totalSec, cssW), totalSec));
            const tA = trimInSec() + tT;
            const newHead = Math.max(trimInSec(), Math.min(tA, trimOutSec()));
            const moved = Math.abs(newHead - runtime.audio.playheadSec) > 1e-4;
            runtime.audio.playheadSec = newHead;
            if (moved) scrubAudio(newHead, 0.12);
            updateAudioUI();
            redraw();
            return;
        }
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);
        const cssW = canvas.clientWidth || 600;
        const { x } = canvasMouse(e);
        const t = xToTime(x, totalSec, cssW);
        const dt = t - runtime.drag.mouseT0;
        const seg = runtime.drag.seg;
        if (runtime.state.segments.indexOf(seg) < 0) { runtime.drag = null; return; }
        const snap = !e.shiftKey;
        const sorted = [...runtime.state.segments].sort((a, b) => a.t_start - b.t_start);
        const idx = sorted.indexOf(seg);

        if (runtime.drag.kind === "body") {
            // Body drag: slide-through-with-swap.
            //   - Dragged seg keeps its size.
            //   - IMMEDIATE neighbors' touching edges follow it (prev's right = seg's left,
            //     next's left = seg's right).
            //   - NON-immediate segments sit at their CANONICAL widths starting from
            //     either end (so they appear unchanged).
            //   - When the dragged seg's pivot crosses a canonical pivot of another seg,
            //     they swap order — the "passed" seg pops back to canonical size on the
            //     other side of the dragged seg.
            const len = runtime.drag.origLen;
            const tentativeStart = runtime.drag.segT0 + dt;
            const tentativePivot = tentativeStart + len / 2;

            const others = runtime.state.segments.filter((s) => s !== seg);
            others.sort((a, b) => runtime.drag.fixedPivots.get(a) - runtime.drag.fixedPivots.get(b));

            let segIdx = 0;
            for (const o of others) {
                if (runtime.drag.fixedPivots.get(o) < tentativePivot) segIdx++;
            }
            const newOrder = [...others.slice(0, segIdx), seg, ...others.slice(segIdx)];

            // Sum canonical widths of NON-immediate segments on each side. (Immediate
            // neighbours are at indices segIdx-1 and segIdx+1 in newOrder.)
            let leftSum = 0;
            for (let i = 0; i < segIdx - 1; i++) {
                leftSum += runtime.drag.fixedWidths.get(newOrder[i]);
            }
            let rightSum = 0;
            for (let i = segIdx + 2; i < newOrder.length; i++) {
                rightSum += runtime.drag.fixedWidths.get(newOrder[i]);
            }
            const hasLeftN  = segIdx > 0;
            const hasRightN = segIdx < newOrder.length - 1;

            // Clamp tentative start so neighbours can't shrink below MIN.
            const minStart = leftSum + (hasLeftN ? MIN_SEG_SEC : 0);
            const maxStart = totalSec - rightSum - len - (hasRightN ? MIN_SEG_SEC : 0);
            let xs = Math.max(minStart, Math.min(tentativeStart, maxStart));
            xs = snapToFrame(xs, fps, snap);
            xs = Math.max(minStart, Math.min(xs, maxStart));
            const xe = xs + len;

            // Layout in newOrder.
            let cursor = 0;
            for (let i = 0; i < newOrder.length; i++) {
                const s = newOrder[i];
                if (s === seg) {
                    s.t_start = xs;
                    s.t_end   = xe;
                    cursor = xe;
                } else if (i === segIdx - 1) {
                    // Immediate left: from cursor (= leftSum after non-imm left) to xs.
                    s.t_start = cursor;
                    s.t_end   = xs;
                    cursor = xs;
                } else if (i === segIdx + 1) {
                    // Immediate right: from xe to (totalSec - rightSum).
                    s.t_start = cursor;
                    s.t_end   = totalSec - rightSum;
                    cursor = s.t_end;
                } else {
                    // Non-immediate: canonical width, tiled from cursor.
                    const w = runtime.drag.fixedWidths.get(s);
                    s.t_start = cursor;
                    s.t_end   = cursor + w;
                    cursor = s.t_end;
                }
            }
        } else if (runtime.drag.kind === "left") {
            // Don't allow crossing the previous segment (regardless of partner status).
            const prev = sorted[idx - 1];
            const minStart = prev ? prev.t_start + MIN_SEG_SEC : 0; // partner can't shrink below MIN
            const partnerCap = runtime.drag.partner
                ? Math.max(runtime.drag.partner.t_start + MIN_SEG_SEC, prev ? prev.t_start + MIN_SEG_SEC : 0)
                : (prev ? prev.t_end : 0);
            let ns = runtime.drag.segT0 + dt;
            ns = Math.max(partnerCap, Math.min(ns, seg.t_end - MIN_SEG_SEC));
            ns = snapToFrame(ns, fps, snap);
            ns = Math.max(partnerCap, Math.min(ns, seg.t_end - MIN_SEG_SEC));
            seg.t_start = ns;
            // Rolling trim: partner (touching prev neighbour) follows our left edge.
            if (runtime.drag.partner) {
                runtime.drag.partner.t_end = ns;
            }
        } else if (runtime.drag.kind === "right") {
            const next = sorted[idx + 1];
            const partnerCap = runtime.drag.partner
                ? Math.min(runtime.drag.partner.t_end - MIN_SEG_SEC, totalSec)
                : (next ? next.t_start : totalSec);
            let ne = runtime.drag.segT1 + dt;
            ne = Math.min(partnerCap, Math.max(ne, seg.t_start + MIN_SEG_SEC));
            ne = snapToFrame(ne, fps, snap);
            ne = Math.min(partnerCap, Math.max(ne, seg.t_start + MIN_SEG_SEC));
            seg.t_end = ne;
            // Rolling trim: partner (touching next neighbour) follows our right edge.
            if (runtime.drag.partner) {
                runtime.drag.partner.t_start = ne;
            }
        }
        commit();
    }
    function onUp() {
        if (runtime.drag) { runtime.drag = null; redraw(); }
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    canvas.addEventListener("keydown", async (e) => {
        const a = runtime.audio;
        const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const k = e.key;

        // Editor-style transport (JKL).
        if (k === "l" || k === "L") {
            if (!a.buffer) return;
            if (a.ctx && a.ctx.state === "suspended") {
                try { await a.ctx.resume(); } catch {}
            }
            const trimEnd = trimOutSec();
            const from = a.playheadSec >= trimEnd - 0.001 ? trimInSec() : a.playheadSec;
            startPlayback(from);
            e.preventDefault();
            return;
        }
        if (k === "k" || k === "K") {
            stopPlayback();
            updateAudioUI();
            redraw();
            e.preventDefault();
            return;
        }
        if (k === "j" || k === "J") {
            if (!a.buffer) return;
            if (a.ctx && a.ctx.state === "suspended") {
                try { await a.ctx.resume(); } catch {}
            }
            const trimStart = trimInSec();
            const from = a.playheadSec <= trimStart + 0.001 ? trimOutSec() : a.playheadSec;
            startPlayback(from, -1);
            e.preventDefault();
            return;
        }

        // In / out points (I / O), set at the current playhead.
        if (k === "i" || k === "I") {
            if (!a.buffer) return;
            const t = a.playheadSec;
            setTrim(t, trimOutSec());
            a.playheadSec = Math.max(trimInSec(), Math.min(a.playheadSec, trimOutSec()));
            if (a.playing) startPlayback(a.playheadSec);
            updateAudioUI();
            redraw();
            e.preventDefault();
            return;
        }
        if (k === "o" || k === "O") {
            if (!a.buffer) return;
            const t = a.playheadSec;
            setTrim(trimInSec(), t);
            a.playheadSec = Math.max(trimInSec(), Math.min(a.playheadSec, trimOutSec()));
            if (a.playing) startPlayback(a.playheadSec);
            updateAudioUI();
            redraw();
            e.preventDefault();
            return;
        }

        // Arrow nudges (audio scrub when audio is loaded).
        if (k === "ArrowLeft" || k === "ArrowRight") {
            if (!a.buffer) return;
            const sign = k === "ArrowRight" ? 1 : -1;
            const step = (e.shiftKey ? 10 : 1) / Math.max(fps, 1);
            a.playheadSec = Math.max(trimInSec(), Math.min(a.playheadSec + sign * step, trimOutSec()));
            if (a.playing) {
                startPlayback(a.playheadSec, a.playDir || 1);
            } else {
                // Audible chunk at the new position — at least 60ms so it's hearable.
                scrubAudio(a.playheadSec, Math.max(0.06, step * 1.2));
            }
            updateAudioUI();
            redraw();
            e.preventDefault();
            return;
        }

        // Tool toggles.
        if (k === "b" || k === "B") {
            setTool(runtime.tool === "blade" ? "select" : "blade");
            e.preventDefault();
            return;
        }
        if (k === "Escape") {
            if (runtime.tool !== "select") {
                setTool("select");
                e.preventDefault();
                return;
            }
        }

        // Delete (existing).
        if ((k === "Delete" || k === "Backspace") && runtime.getSelected()) {
            const sel = runtime.getSelected();
            runtime.state.segments = runtime.state.segments.filter((s) => s !== sel);
            runtime.selectedRef = null;
            rescaleToFill(runtime.state.segments, totalSec);
            commit();
            e.preventDefault();
        }
    });

    canvas.addEventListener("contextmenu", (e) => {
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const { x, y } = canvasMouse(e);
        const hit = hitTest(runtime.state, totalSec, canvas.clientWidth || 600, x, y);
        if (hit) {
            const seg = runtime.state.segments[hit.index];
            runtime.state.segments = runtime.state.segments.filter((s) => s !== seg);
            if (runtime.selectedRef === seg) runtime.selectedRef = null;
            rescaleToFill(runtime.state.segments, totalSec);
            commit();
            e.preventDefault();
        }
    });

    // audio_file widget: load the file when changed. ComfyUI's upload widget
    // doesn't always invoke widget.callback on selection, so we ALSO poll the
    // value at a low rate as a safety net.
    {
        const w = node.widgets?.find((x) => x.name === "audio_file");
        if (w) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                loadAudio(w.value);
                return r;
            };
        }
        const pollIv = setInterval(() => {
            const w2 = node.widgets?.find((x) => x.name === "audio_file");
            if (!w2) return;
            const v = w2.value;
            const cur = runtime.audio.currentFile;
            if (v && v !== "(none)" && v !== cur) {
                loadAudio(v);
            } else if ((!v || v === "(none)") && cur) {
                loadAudio(null);
            }
        }, 400);
        // Stop polling if the node is removed.
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function (...a) {
            clearInterval(pollIv);
            return origOnRemoved?.apply(this, a);
        };
    }

    // total_duration_sec changes: rescale all segments proportionally to fit the new total.
    {
        const w = getNumberWidget(node, "total_duration_sec");
        if (w) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                const newTotal = Number(w.value ?? 4);
                if (runtime.state.segments.length > 0 && newTotal > 0) {
                    rescaleToFill(runtime.state.segments, newTotal);
                    writeState(node, runtime.state);
                }
                redraw();
                return r;
            };
        }
    }
    // frame_rate changes: just redraw (tick density may change).
    {
        const w = getNumberWidget(node, "frame_rate");
        if (w) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                redraw();
                return r;
            };
        }
    }

    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = origConfigure?.apply(this, args);
        requestAnimationFrame(() => {
            hideTimelineDataWidget(node);
            runtime.state = readState(node);
            styleArea.value = runtime.state.global || "";
            runtime.selectedRef = null;
            redraw();
        });
        return r;
    };

    // Initial sizing pass + first render
    requestAnimationFrame(() => {
        hideTimelineDataWidget(node);
        redraw();
    });

    node._rsTimeline = { canvas, redraw };
}

app.registerExtension({
    name: "rs-nodes.PromptRelayTimeline",
    nodeCreated(node) {
        if (node.comfyClass === "RSPromptRelayTimeline") {
            attachTimeline(node);
        }
    },
});
