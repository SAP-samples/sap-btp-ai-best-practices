/**
 * Route Configuration
 */

export const routes = [
  // Home page (mapped to root and /dashboard)
  "dashboard",

  // Main application pages
  "orders",
  "fine-tuning"
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/": "/orders",
  "/home": "/orders"
};
