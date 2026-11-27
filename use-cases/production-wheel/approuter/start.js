/**
 * Start the SAP application router with the routes of the chosen deployment.
 *
 * The same approuter code serves two deployments of the Production Wheel
 * Optimizer (see docs/user-authentication.md, "Private and public deployments"):
 *
 *   - private (manifest.yaml, ./deploy.sh private): routes from xs-app.json.
 *     Users sign in through SAP BTP (XSUAA) and need the role collection
 *     ProductionWheel_User.
 *   - public (manifest-public.yaml, ./deploy.sh public): routes from
 *     xs-app.public.json. No sign-in; for showing the app externally.
 *
 * The XS_APP_CONFIG environment variable names the routes file. It defaults to
 * xs-app.json, so a deployment that does not set it always gets sign-in.
 *
 * Usage (normally run by "npm start" in Cloud Foundry):
 *   node start.js
 *   XS_APP_CONFIG=xs-app.public.json node start.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const approuter = require("@sap/approuter");

// Only these files may be selected, so a wrong value can never load an
// arbitrary file; anything else stops the start with a clear error.
const ALLOWED_CONFIGS = ["xs-app.json", "xs-app.public.json"];
const DEFAULT_CONFIG = "xs-app.json";

/**
 * Read the routes file named by XS_APP_CONFIG.
 *
 * @param {string | undefined} requested Value of XS_APP_CONFIG (may be empty).
 * @returns {object} Parsed xs-app configuration for approuter.start().
 * @throws {Error} When the name is not one of ALLOWED_CONFIGS.
 */
function loadRoutes(requested) {
  const name = (requested || DEFAULT_CONFIG).trim();
  if (!ALLOWED_CONFIGS.includes(name)) {
    throw new Error(`XS_APP_CONFIG must be one of ${ALLOWED_CONFIGS.join(", ")}; got "${name}"`);
  }
  console.log(`Application router routes: ${name}`);
  return JSON.parse(fs.readFileSync(path.join(__dirname, name), "utf8"));
}

approuter().start({ xsappConfig: loadRoutes(process.env.XS_APP_CONFIG) });
