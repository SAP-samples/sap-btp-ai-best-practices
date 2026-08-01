sap.ui.define([
    "sap/ui/core/UIComponent",
    "financial/flux/analysis/model/models",
    "financial/flux/analysis/util/formatter"
], function (UIComponent, models, formatter) {
    "use strict";

    return UIComponent.extend("financial.flux.analysis.Component", {
        metadata: {
            manifest: "json"
        },
        init: function () {
            UIComponent.prototype.init.apply(this, arguments);
            this.getRouter().initialize();
            var oModel = models.createDeviceModel();
            this.setModel(oModel, "device");
        }
    });
});