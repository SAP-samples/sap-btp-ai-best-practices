sap.ui.define(["sap/m/MessageBox", "sap/ui/core/Fragment"],
    function (MessageBox, Fragment) {
        "use strict";

        return {
            onUpload: function (oBindingContext, aSelectedContexts) {
                const oView = this._view;
                const sUploadUrl = `${oView.getModel().getServiceUrl()}ScienceDataUpload/content`;
                const oUploadModel = new sap.ui.model.json.JSONModel({ fileName: "", uploadUrl: sUploadUrl });
                oView.setModel(oUploadModel, "uploadModel");

                // Show Dialog
                if (!this.oUploadDialog) {
                    Fragment.load({
                        name: "ui5.manage.sciencedata.ext.view.Upload",
                        controller: this
                    })
                        .then(function (oDialog) {
                            oView.oUploadDialog = oDialog;
                            oView.addDependent(oDialog);
                            oDialog.open();
                        });
                }
                else {
                    oView.oUploadDialog.open();
                }
            },

            onCancel: function (oEvent) {
                this._view.oUploadDialog.close();
                this._view.oUploadDialog.destroy();
            },

            onSave: function (oEvent) {
                const oFileUploader = sap.ui.getCore().byId("fileUploader");
                const oDialog = this._view.oUploadDialog;

                oDialog.setBusy(true);

                // CSRF TOKEN
                const oHeaders = this._view.getModel().getHttpHeaders();
                if ('X-CSRF-Token' in oHeaders) {
                    var headerParam = new sap.ui.unified.FileUploaderParameter({ name: 'X-CSRF-Token', value: oHeaders['X-CSRF-Token'] });
                    oFileUploader.addHeaderParameter(headerParam);
                }

                // Call Upload
                oFileUploader.checkFileReadable().then(function () {
                    oFileUploader.upload();
                }, function (error) {
                    MessageBox.show("The file cannot be read. It may have changed.");
                });

            },

            onUploadComplete: function (oEvent) {
                const oDialog = this._view.oUploadDialog;
                const oResponseRaw = oEvent.getParameter("responseRaw");

                const iStatus = oEvent.getParameter("status");

                oDialog.setBusy(false);

                if (iStatus === 204) {
                    MessageBox.show("Upload Sucess!  \nClick on 'Go' to refresh the list report.");
                    this._view.oUploadDialog.close();
                    this._view.oUploadDialog.destroy();

                } else {
                    MessageBox.error(`Upload Error: ${oResponseRaw}`);
                }
            }
        }
    });
