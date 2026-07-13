#!/usr/bin/env node

/**
 * Page Generator Script
 *
 * This script helps create new page templates quickly.
 * Usage: node scripts/create-page.js <pageName> [title]
 *
 * Example: node scripts/create-page.js products "Product Catalog"
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function createPage(pageName, title = null) {
  if (!pageName) {
    console.error("❌ Error: Page name is required");
    console.log("Usage: node scripts/create-page.js <pageName> [title]");
    console.log('Example: node scripts/create-page.js products "Product Catalog"');
    process.exit(1);
  }

  const pageTitle = title || pageName.charAt(0).toUpperCase() + pageName.slice(1);
  const pagesDir = path.join(__dirname, "../src/pages");
  const pageDir = path.join(pagesDir, pageName);

  // Check if page already exists
  if (fs.existsSync(pageDir)) {
    console.error(`❌ Error: Page "${pageName}" already exists`);
    process.exit(1);
  }

  // Create page directory
  fs.mkdirSync(pageDir, { recursive: true });

  // HTML template
  const htmlContent = `<ui5-title>${pageTitle}</ui5-title>
<br />
<ui5-text>
  This is the ${pageTitle} page. Add your content here.
</ui5-text>

<!-- Add your page content below -->
<div class="${pageName}-content">
  <!-- Your page-specific UI components go here -->
</div>`;

  // JavaScript template
  const jsContent = `/* ${pageTitle} page specific UI5 components */
// Import any UI5 components used only on this page
// Example: import "@ui5/webcomponents/dist/Button.js";

/* ${pageTitle} page specific icons - these will load before HTML rendering */
// Import any UI5 icons used only on this page
// Example: import "@ui5/webcomponents-icons/dist/add.js";

export default function init${pageTitle.replace(/\s+/g, "")}Page() {
  console.log('${pageTitle} page initialized');
  
  // Add your page-specific functionality here
  // Example:
  // const button = document.getElementById('my-button');
  // if (button) {
  //   button.addEventListener('click', () => {
  //     console.log('Button clicked!');
  //   });
  // }
}`;

  // CSS template
  const cssContent = `/* ${pageTitle} page specific styles */
.${pageName}-container {
  padding: 1rem;
  max-width: 1200px;
  margin: 0 auto;
}

.${pageName}-content {
  margin-top: 1rem;
}

.${pageName}-section {
  margin-bottom: 2rem;
}

/* Add your page-specific styles here */`;

  // Write files to page directory
  const htmlFile = path.join(pageDir, `${pageName}.html`);
  const jsFile = path.join(pageDir, `${pageName}.js`);
  const cssFile = path.join(pageDir, `${pageName}.css`);

  fs.writeFileSync(htmlFile, htmlContent);
  fs.writeFileSync(jsFile, jsContent);
  fs.writeFileSync(cssFile, cssContent);

  console.log(`✅ Page "${pageName}" created successfully!`);
  console.log(`📁 Files created:`);
  console.log(`   ${htmlFile}`);
  console.log(`   ${jsFile}`);
  console.log(`   ${cssFile}`);
  // Auto-add route to configuration
  addRouteToConfig(pageName);

  console.log(`\n📋 Next steps:`);
  console.log(`1. ✅ Route automatically added to src/config/routes.js`);
  console.log(`2. Add a navigation item in your index.html:`);
  console.log(`   <ui5-side-navigation-item text="${pageTitle}" href="/${pageName}" icon="your-icon"></ui5-side-navigation-item>`);
  console.log(`3. Test your page by navigating to: http://localhost:5173/${pageName}`);
  console.log(`4. Customize the HTML, JS, and CSS files as needed`);
}

function addRouteToConfig(pageName) {
  try {
    const routesFile = path.join(__dirname, "../src/config/routes.js");
    let routesContent = fs.readFileSync(routesFile, "utf8");

    // Check if route already exists
    if (routesContent.includes(`'${pageName}'`)) {
      console.log(`ℹ️ Route '${pageName}' already exists in configuration`);
      return;
    }

    // Find the routes array and add the new route
    const routesArrayMatch = routesContent.match(/(export const routes = \[[\s\S]*?)(  \/\/ Example of advanced)/);

    if (routesArrayMatch) {
      const beforeExample = routesArrayMatch[1];
      const afterExample = routesArrayMatch[2];

      const newRoutesContent = routesContent.replace(routesArrayMatch[0], `${beforeExample}  '${pageName}',\n\n  ${afterExample}`);

      fs.writeFileSync(routesFile, newRoutesContent);
      console.log(`✅ Added '${pageName}' to src/config/routes.js`);
    } else {
      console.log(`⚠️ Could not automatically add route. Please add '${pageName}' to src/config/routes.js manually`);
    }
  } catch (error) {
    console.log(`⚠️ Could not update routes file: ${error.message}`);
    console.log(`Please add '${pageName}' to src/config/routes.js manually`);
  }
}

// Get command line arguments
const args = process.argv.slice(2);
const pageName = args[0];
const title = args[1];

createPage(pageName, title);
