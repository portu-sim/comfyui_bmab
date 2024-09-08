import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.BMAB.BMABRemoteAccessAndSave",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BMAB Remote Access And Save") {

            function register_client_id(name) {
                fetch(api.apiURL(`bmab?remote_client_id=${api.clientId}&remote_name=${name}`))
                    .then(response => response.json())
                    .then(data => {
                        console.log(data);
                    })
                    .catch(error => {
                        console.error(error);
                    });
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const remote_name = this.widgets.find((w) => w.name === "remote_name");
                register_client_id(remote_name.value);

                remote_name.callback = function () {
                    register_client_id(this.value);
                }
            };

            const onReconnect = nodeType.prototype.onReconnect;
            nodeType.prototype.onReconnect = function () {
                const remote_name = this.widgets.find((w) => w.name === "remote_name");
                register_client_id(remote_name.value);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (w) {
                const remote_name = this.widgets.find((w) => w.name === "remote_name");
                register_client_id(remote_name.value);
            };

            const onBMABQueue = nodeType.prototype.onBMABQueue;
            nodeType.prototype.onBMABQueue = function () {
                console.log('QUEUE prompt')
                app.queuePrompt(0, 1)
            };

            api.addEventListener("reconnected", ({ detail }) => {
                app.graph._nodes.forEach((node) => {
                    if (node.onReconnect)
                        node.onReconnect()
                })
            });

            api.addEventListener("bmab_queue", ({ detail }) => {
                app.graph._nodes.forEach((node) => {
                    if (node.onBMABQueue)
                        node.onBMABQueue()
                })
            });
        }
    },
});
