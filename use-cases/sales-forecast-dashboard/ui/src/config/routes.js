/**
 * Route Configuration
 *
 * Add your page routes here. The router will automatically register these routes.
 *
 * Format:
 * - string: Simple route that maps to a page with the same name
 * - object: Advanced route configuration with custom settings
 */

export const routes = [
  // Dashboard page (main page)
  "dashboard",

  // AI Chatbot page
  "chatbot",

  // Scenario Maker page
  "scenario-maker"
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/home": "/dashboard"
};
