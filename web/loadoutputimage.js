// this code from ComfyUI_Custom_Nodes_AlekPet

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.BMAB.LoadOutputImage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BMAB Load Output Image") {

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (texts) {
                onConfigure?.apply(this, arguments);

                const node = this
                const imageWidget = node.widgets.find((w) => w.name === "image");
                function showImage(name) {
                    const img = new Image();
                    img.onload = () => {
                        node.imgs = [img];
                        app.graph.setDirtyCanvas(true);
                    };
                    img.src = api.apiURL(`/view?filename=${name}&type=output${app.getRandParam()}`);
                    node.setSizeForImage?.();
                }

                const cb = this.callback;
                imageWidget.callback = function () {
                    showImage(imageWidget.value);
                    app.graph.setDirtyCanvas(true);
                    if (cb) {
                        return cb.apply(this, arguments);
                    }
                };

                showImage(imageWidget.value);
                app.graph.setDirtyCanvas(true);
            };
        }
    },
});
