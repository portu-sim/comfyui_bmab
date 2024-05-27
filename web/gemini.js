import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.BMAB.GoogleGeminiPromptNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BMAB Google Gemini Prompt") {

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (texts) {
                console.log(`on executed ${texts}`);
                onExecuted?.apply(this, arguments);
                let widget_id = this?.widgets.findIndex(
                    obj => obj.name === 'random_seed'
                );
                console.log(`on set ${texts?.string}`);
                this.widgets[widget_id].value = Number(texts?.string)
                app.graph.setDirtyCanvas(true);
            };
        }
    },
});
