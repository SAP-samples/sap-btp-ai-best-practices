sap.ui.define([
    "sap/m/MessageToast"
], function(MessageToast) {
    'use strict';

    return {
        /**
         * Generated event handler.
         *
         * @param oContext the context of the page on which the event was fired. `undefined` for list report page.
         * @param aSelectedContexts the selected contexts of the table rows.
         */
        syncDocuments: async function(oContext, aSelectedContexts) {
            MessageToast.show("Synchronizing extracted documents...");

            const oOp = this.getModel().bindContext("/SyncDOX(...)");
            const result = await oOp.execute();
            console.log(result)
            MessageToast.show("Synchronization done.");        
        },

        generateMappings: async function() {
            MessageToast.show("Mapping Purchase Orders to Sales Orders...");
            const oOp = this.getModel().bindContext("/generateMapping(...)");
            const result = await oOp.execute();
            console.log(result)
            MessageToast.show("Mapping done."); 
        }
    };
});
