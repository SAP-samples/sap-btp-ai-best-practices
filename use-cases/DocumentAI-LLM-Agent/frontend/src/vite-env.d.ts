/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_APP_TITLE: string;
  readonly VITE_APP_SUBTITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
// UI5 module declarations
declare module "@ui5/webcomponents-base/dist/config/Theme.js" {
  export function setTheme(theme: string): void;
  export function getTheme(): string;
}
declare module "@ui5/webcomponents/dist/*.js" { const c: unknown; export default c; }
declare module "@ui5/webcomponents-fiori/dist/*.js" { const c: unknown; export default c; }
declare module "@ui5/webcomponents-icons/dist/*.js" { const c: unknown; export default c; }
declare module "@ui5/webcomponents/dist/Assets.js" {}
declare module "@ui5/webcomponents-fiori/dist/Assets.js" {}
declare module "@ui5/webcomponents-icons/dist/AllIcons.js" {}
declare module "@ui5/webcomponents/dist/Button.js" {}
declare module "@ui5/webcomponents/dist/Card.js" {}
declare module "@ui5/webcomponents/dist/CardHeader.js" {}
declare module "@ui5/webcomponents/dist/BusyIndicator.js" {}
declare module "@ui5/webcomponents/dist/MessageStrip.js" {}
declare module "@ui5/webcomponents/dist/Label.js" {}
declare module "@ui5/webcomponents/dist/Title.js" {}
declare module "@ui5/webcomponents/dist/Icon.js" {}
declare module "@ui5/webcomponents-fiori/dist/ShellBar.js" {}
declare module "@ui5/webcomponents-fiori/dist/ShellBarItem.js" {}
