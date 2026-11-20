#!/usr/bin/env node

/**
 * Page Generator Script
 *
 * Creates a new React page (component + stylesheet) and registers it in src/routes.tsx,
 * which automatically adds the route and the side navigation item.
 *
 * Usage (from the ui/ directory):
 *   npm run create-page <pageName> ["Page Title"] [icon]
 *   node scripts/create-page.js <pageName> ["Page Title"] [icon]
 *
 * Examples:
 *   npm run create-page products "Product Catalog" product
 *   npm run create-page settings "User Settings" action-settings
 *   npm run create-page reports
 *
 * Arguments:
 *   pageName  kebab-case or lower-case folder/URL name, e.g. "sales-report" -> /sales-report
 *   title     label in the side navigation (default: derived from pageName)
 *   icon      UI5 icon name (default: "document")
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC_DIR = path.join(__dirname, "../src");
const ROUTES_FILE = path.join(SRC_DIR, "routes.tsx");
const ROUTES_END_MARKER = "// create-page:routes-end";

/**
 * Converts "sales-report" to "SalesReport".
 *
 * @param {string} name - kebab-case name.
 * @returns {string} PascalCase name.
 */
function toPascalCase(name) {
  return name
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");
}

/**
 * Builds the page component source.
 *
 * @param {string} componentName - PascalCase component name.
 * @param {string} pageName - kebab-case page name (CSS class prefix).
 * @param {string} title - Page title.
 * @returns {string} TSX source.
 */
function pageTemplate(componentName, pageName, title) {
  return `/**
 * ${title} page.
 */
import { Text, Title } from "@ui5/webcomponents-react";

import "./${pageName}.css";

/**
 * @returns The ${title} page.
 */
export default function ${componentName}() {
  return (
    <div className="${pageName}-container">
      <Title>${title}</Title>
      <br />
      <Text>This is the ${title} page. Add your content here.</Text>

      <div className="${pageName}-content">{/* Your page-specific UI components go here */}</div>
    </div>
  );
}
`;
}

/**
 * Builds the page stylesheet.
 *
 * @param {string} pageName - kebab-case page name (CSS class prefix).
 * @param {string} title - Page title.
 * @returns {string} CSS source.
 */
function cssTemplate(pageName, title) {
  return `/* ${title} page specific styles */
.${pageName}-container {
  padding: 1rem;
}

.${pageName}-content {
  margin-top: 1rem;
}
`;
}

/**
 * Creates the page files and registers the route.
 *
 * @param {string} pageName - kebab-case page name.
 * @param {string | undefined} title - Optional page title.
 * @param {string | undefined} icon - Optional UI5 icon name.
 */
function createPage(pageName, title, icon = "document") {
  if (!pageName || !/^[a-z][a-z0-9-]*$/.test(pageName)) {
    console.error("Error: pageName is required and must be lower-case/kebab-case (e.g. sales-report)");
    console.log('Usage: npm run create-page <pageName> ["Page Title"] [icon]');
    process.exit(1);
  }

  const componentName = `${toPascalCase(pageName)}Page`;
  const pageTitle = title || toPascalCase(pageName).replace(/([a-z])([A-Z])/g, "$1 $2");
  const pageDir = path.join(SRC_DIR, "pages", pageName);

  if (fs.existsSync(pageDir)) {
    console.error(`Error: page "${pageName}" already exists at ${pageDir}`);
    process.exit(1);
  }

  const routesSource = fs.readFileSync(ROUTES_FILE, "utf8");
  if (!routesSource.includes(ROUTES_END_MARKER)) {
    console.error(`Error: marker "${ROUTES_END_MARKER}" not found in src/routes.tsx`);
    process.exit(1);
  }

  fs.mkdirSync(pageDir, { recursive: true });
  fs.writeFileSync(path.join(pageDir, `${componentName}.tsx`), pageTemplate(componentName, pageName, pageTitle));
  fs.writeFileSync(path.join(pageDir, `${pageName}.css`), cssTemplate(pageName, pageTitle));

  // Insert the new route right before the end marker, keeping the marker's indentation.
  const routeLine =
    `  { path: "/${pageName}", title: ${JSON.stringify(pageTitle)}, icon: ${JSON.stringify(icon)}, ` +
    `Component: lazy(() => import("./pages/${pageName}/${componentName}")) },\n`;
  fs.writeFileSync(ROUTES_FILE, routesSource.replace(`  ${ROUTES_END_MARKER}`, `${routeLine}  ${ROUTES_END_MARKER}`));

  console.log(`Created page "${pageTitle}"`);
  console.log(`  src/pages/${pageName}/${componentName}.tsx`);
  console.log(`  src/pages/${pageName}/${pageName}.css`);
  console.log(`  route /${pageName} registered in src/routes.tsx (side navigation updates automatically)`);
}

const [, , pageName, title, icon] = process.argv;
createPage(pageName, title, icon);
