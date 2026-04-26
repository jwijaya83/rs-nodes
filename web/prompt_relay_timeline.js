// RSPromptRelayTimeline — canvas timeline widget.
//
// Step G1: skeleton only. Subsequent steps add the canvas, interactions,
// inline editing, etc. See Reference/prompt-relay-timeline-plan.md.

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "rs-nodes.PromptRelayTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RSPromptRelayTimeline") return;
        // Future steps will hook nodeType.prototype.onNodeCreated here to
        // inject the canvas widget and wire it to the timeline_data STRING.
    },
});
