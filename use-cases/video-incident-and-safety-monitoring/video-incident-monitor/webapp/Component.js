sap.ui.define(
    ["sap/ui/core/UIComponent", "sap/ui/model/json/JSONModel"],
    function (UIComponent, JSONModel) {
        "use strict";

        return UIComponent.extend("videoincidentmonitor.Component", {
            metadata: {
                manifest: "json"
            },

            init: function () {
                // Call the base component's init function
                UIComponent.prototype.init.apply(this, arguments);

                // Create the views based on the url/hash
                this.getRouter().initialize();
            }
        });
    }
);
