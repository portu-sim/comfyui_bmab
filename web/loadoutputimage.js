// this code from ComfyUI_Custom_Nodes_AlekPet

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.BMAB.LoadOutputImage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BMAB Load Output Image") {


            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this
                function showImage(name) {
                    const img = new Image();
                    img.onload = () => {
                        node.imgs = [img];
                        app.graph.setDirtyCanvas(true);
                    };

                    const split = name.split('/');
                    let subdir = ''
                    let fileName = ''
                    if (split.length === 1) {
                        fileName = split[0];
                    } else {
                        subdir = split.slice(0, split.length - 1).join('/');
                        fileName = split[split.length - 1];
                    }
                    img.src = api.apiURL(`/view?filename=${fileName}&subfolder=${subdir}&type=output${app.getRandParam()}`);
                    node.setSizeForImage?.();
                }

                const imageWidget = node.widgets.find((w) => w.name === "image");

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
