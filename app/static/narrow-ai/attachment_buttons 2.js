sap.ui.define([
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/BusyIndicator",
    "sap/m/Dialog",
    "sap/m/VBox",
    "sap/m/Text",
    "sap/m/Button",
    "sap/m/ProgressIndicator"
], function (MessageToast, MessageBox, Busy, Dialog, VBox, Text, Button, ProgressIndicator) {
    "use strict";

    // =========================
    // Utilities
    // =========================

    /**
     * Helper to fetch i18n texts from the resource bundle.
     *
     * @param {sap.ui.base.ManagedObject} that - The current controller context (usually "this").
     * @returns {(key:string, args?:any[]) => string} Function that resolves i18n keys with optional arguments.
     */
    function getI18n(that) {
        const oBundle = that.getModel("i18n")?.getResourceBundle();
        return (key, args) => (oBundle ? oBundle.getText(key, args) : key);
    }

    /**
     * Execute a bound OData V4 action on the current context.
     * @param {sap.ui.model.odata.v4.Context} oCtx - Bound context of the OP.
     * @param {string} name - Action name (w/o namespace).
     * @param {object} [params] - Named parameters for the action.
     * @returns {Promise<any>} Action result object (already unwrapped from the bound context).
     */
    async function callAction(oCtx, name, params) {
        const oModel = oCtx.getModel();
        const path = `${oCtx.getPath()}/BudgetService.${name}(...)`;
        const act = oModel.bindContext(path);
        Object.entries(params || {}).forEach(([k, v]) => act.setParameter(k, v));
        await act.execute();
        return act.getBoundContext()?.getObject();
    }

    /**
     * Read a File object and return its Base64 payload (no data: prefix).
     * @param {File} file
     * @returns {Promise<string>} Base64 content (payload only).
     */
    function encodeB64(file) {
        return new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(String(r.result).split(",")[1] || "");
            r.onerror = reject;
            r.readAsDataURL(file);
        });
    }

    /**
     * Approximate the number of bytes represented by a Base64 chunk.
     * @param {string} b64
     * @returns {number} Approx. byte count.
     */
    function approxBytesFromB64(b64) {
        let pad = 0;
        if (b64.endsWith("==")) pad = 2;
        else if (b64.endsWith("=")) pad = 1;
        return Math.floor(b64.length * 3 / 4) - pad;
    }

    /** Pretty-print bytes. @param {number} n */
    function fmtBytes(n) {
        if (n < 1024) return `${n} B`;
        if (n < 1024 * 1024) return `${(n / 1024).toFixed(2)} KB`;
        return `${(n / 1024 / 1024).toFixed(2)} MB`;
    }

    /** Pretty-print seconds. @param {number} sec */
    function fmtTime(sec) {
        if (!isFinite(sec) || sec < 0) return "—";
        if (sec < 60) return `${Math.round(sec)}s`;
        const m = Math.floor(sec / 60), s = Math.round(sec % 60);
        return `${m}m ${s}s`;
    }

    /**
     * Build localized texts for the upload dialog.
     * Put these keys in i18n/i18n.properties:
     *
     * uploadDialog.title=Uploading {0}
     * uploadDialog.waiting=Waiting...
     * uploadDialog.finishing=Finishing upload…
     * common.cancel=Cancel
     * uploadDialog.progress={0} / {1} • {2}/s • ETA {3}
     *
     * @param {( key:string, args?:any[]) => string} i18n - i18n resolver (getI18n(this)).
     * @param { string } fileLabel - File label (e.g., "KEYID.ZIP") to interpolate into the title.
     * @returns {{ title:string, waiting:string, finishing:string, cancel:string }}
     */
    function makeUploadDialogTexts(i18n, fileLabel) {
        return {
            title: i18n("uploadDialog.title", [fileLabel]),
            waiting: i18n("uploadDialog.waiting"),
            finishing: i18n("uploadDialog.finishing"),
            cancel: i18n("common.cancel")
        };
    }

    /**
     * Create a compact progress dialog with a cancel button.
     * All user-facing strings are supplied via `texts`.
     *
     * @param {{ onCancel?: Function }} [opts] - Optional callbacks.
     * @param {{ title:string, waiting:string, finishing:string, cancel:string }} texts - Localized texts.
     */
    function createProgressDialog(opts, texts) {
        let cancelled = false;
        let finalizing = false;

        const oProgress = new ProgressIndicator({
            percentValue: 0,
            showValue: true,
            width: "100%",
            displayValue: "0%"
        }).addStyleClass("sapUiTinyMarginTopBottom");

        const oText = new Text({ text: texts.waiting })
        .addStyleClass("sapUiTinyMarginTopBottom");

        const oCancelBtn = new Button({
            text: texts.cancel,
            type: "Transparent",
            press: async function () {
                // ignore clicks once finalization started
                if (finalizing) return;
                cancelled = true;
                try { await opts?.onCancel?.(); } catch { /* no-op */ }
                oDialog.close();
            }
        });

        const oDialog = new Dialog({
            title: texts.title,
            type: "Message",
            contentWidth: "420px",
            content: new VBox({ width: "100%", items: [oProgress, oText] })
            .addStyleClass("sapUiSmallMargin"),
            buttons: [oCancelBtn],
            afterClose: function () { oDialog.destroy(); }
        });

        return {
            /** Open dialog. */
            open: () => oDialog.open(),
            /** Close dialog. */
            close: () => oDialog.close(),
            /** @returns {boolean} Whether the user has cancelled. */
            isCancelled: () => cancelled,

            /**
             * Update progress bar + footer text (speed/ETA).
             * @param {number} sentBytes
             * @param {number} totalBytes
             * @param {number} elapsedMs
             */
            update: (sentBytes, totalBytes, elapsedMs) => {
                const percent = Math.max(0, Math.min(100, Math.round(sentBytes * 100 / Math.max(1, totalBytes))));
                const speed = sentBytes / Math.max(1, elapsedMs / 1000); // B/s
                const eta = speed > 0 ? (totalBytes - sentBytes) / speed : Infinity;

                oProgress.setPercentValue(percent);
                oProgress.setDisplayValue(`${percent}%`);

                oText.setText(
                    ("{0} / {1} • {2}/s • ETA {3}")
                    .replace("{0}", fmtBytes(sentBytes))
                    .replace("{1}", fmtBytes(totalBytes))
                    .replace("{2}", fmtBytes(speed))
                    .replace("{3}", fmtTime(eta))
                );
            },

            /**
             * Lock UI and show “Finishing…” while `finishUploadZip` is in flight.
             * @param {string} [msg] - Optional custom message.
             */
            startFinalizing: (msg) => {
                finalizing = true;
                oCancelBtn.setVisible(false);
                oProgress.setPercentValue(100);
                oProgress.setDisplayValue("100%");
                oText.setText(msg || texts.finishing);
            }
        };
    }

    // =========================
    // Main flows
    // =========================

    /**
     * Upload a file as Base64 in chunks via three bound actions:
     * 1) startUploadZip  2) uploadZipChunk (N times)  3) finishUploadZip.
     *
     * @param {sap.ui.model.odata.v4.Context} oCtx
     * @param {File} file
     * @param {{ title:string, waiting:string, finishing:string, cancel:string }} texts - Localized texts.
     */
    async function uploadInChunks(oCtx, file, texts, that) {
        const i18n = getI18n(that);

        // Size of each Base64 chunk (characters, not bytes)
        const B64_CHUNK = 80 * 1024;

        const dlg = createProgressDialog({
            onCancel: async () => {
                try { await callAction(oCtx, "cancelUploadZip", {}); } catch { /* ignore */ }
            }
        }, texts);
        dlg.open();

        try {
            // 1) Read file to Base64 (single string, then chunk it locally)
            const full = await encodeB64(file);

            // 2) Initialize server-side buffer
            await callAction(oCtx, "startUploadZip", { fileName: file.name });

            // 3) Stream chunks
            let sentBytes = 0;
            let seq = 0;
            const totalBytes = file.size;
            const t0 = Date.now();

            for (let i = 0; i < full.length; i += B64_CHUNK) {
                if (dlg.isCancelled()) throw new Error(i18n("UploadCancelled"));

                const chunkB64 = full.slice(i, i + B64_CHUNK);
                await callAction(oCtx, "uploadZipChunk", { seq, chunkB64 });
                seq++;

                sentBytes += approxBytesFromB64(chunkB64);
                dlg.update(Math.min(sentBytes, totalBytes), totalBytes, Date.now() - t0);
            }

            // 4) Finalize (hide Cancel, keep dialog until the call returns)
            dlg.startFinalizing(); // text comes from texts.finishing

            await callAction(oCtx, "finishUploadZip", {
                keyID: oCtx.getObject("keyID"),
                fileName: file.name
            });

            MessageToast.show(i18n("ZIPUploaded"));
            await oCtx.requestRefresh();
        } catch (e) {
            if (e.message !== i18n("UploadCancelled")) {
                MessageBox.error(e?.message || i18n("UploadFailed"));
                // eslint-disable-next-line no-console
                console.error("[Upload] failed:", e);
            }
        } finally {
            dlg.close();
        }
    }

    /**
     * Download a ZIP via bound action returning Base64.
     * @param {sap.ui.model.odata.v4.Context} oCtx
     */
    async function downloadZip(oCtx, that) {
        const i18n = getI18n(that);
        Busy.show(0);
        try {
            const keyID = oCtx.getObject("keyID");
            const fileName = oCtx.getObject("FileName");

            if (!keyID) {
                MessageBox.error(i18n("MissingKeyIDInContext"));
                return;
            }

            const raw = await callAction(oCtx, "downloadZip", { keyID, fileName });

            // OData V4 primitive may come as { value: "..." }
            let b64 =
                (typeof raw === "string" && raw) ||
                (raw && typeof raw.value === "string" && raw.value) ||
                (raw && typeof raw.EV_FILE === "string" && raw.EV_FILE) ||
                "";

            // Normalize: remove data: prefix and whitespace/newlines
            b64 = b64.replace(/^data:.*;base64,/, "").replace(/\s/g, "");
            if (!b64) {
                MessageBox.error(i18n("NoFileContentReceived"));
                return;
            }

            // Use data URL + fetch → robust Blob creation (avoids atob issues)
            const resp = await fetch(`data:application/zip;base64,${b64}`);
            const blob = await resp.blob();

            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
        } catch (e) {
            MessageBox.error(e?.message || i18n("DownloadFailed"));
            // eslint-disable-next-line no-console
            console.error("[ext.ProjectOP] Download failed:", e);
        } finally {
            Busy.hide();
        }
    }

    // =========================
    // Controller API
    // =========================

    return {
        /**
         * Header action handler: select ZIP and upload in chunks.
         */
        onUpload: async function () {
            const oCtx = this.getBindingContext();

            // Open file picker
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".zip,application/zip";
            input.click();
            await new Promise(r => input.onchange = r);

            const file = input.files?.[0];
            if (!file) return;

            // Build localized texts for the dialog
            const i18n = getI18n(this);
            const fileLabel = file.name;
            const texts = makeUploadDialogTexts(i18n, fileLabel);

            await uploadInChunks(oCtx, file, texts, this);
        },

        /**
         * Header action handler: download ZIP for current keyID.
         */
        onDownload: async function () {
            const oCtx = this.getBindingContext();
            await downloadZip(oCtx, this);
        }
    };
});