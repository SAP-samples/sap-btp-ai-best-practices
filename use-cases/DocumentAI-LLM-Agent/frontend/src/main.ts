// SAP UI5 Web Components — Horizon Theme
import "@ui5/webcomponents/dist/Assets.js";
import "@ui5/webcomponents-fiori/dist/Assets.js";
import "@ui5/webcomponents-icons/dist/AllIcons.js";

// Core components
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Icon.js";

// Fiori components
import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/ShellBarItem.js";

// Theme: Horizon
import { setTheme } from "@ui5/webcomponents-base/dist/config/Theme.js";
setTheme("sap_horizon");

// App
import { App } from "./components/app.js";

const root = document.getElementById("app");
if (!root) throw new Error("Root element #app not found");

const app = new App(root);
app.init().catch(console.error);
