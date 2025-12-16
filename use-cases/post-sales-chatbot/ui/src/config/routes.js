/**
 * Route Configuration for Apex Automotive Services UI
 *
 * Add your page routes here. The router will automatically register these routes.
 *
 * Format:
 * - string: Simple route that maps to a page with the same name
 * - object: Advanced route configuration with custom settings
 */

export const routes = [
  // Home page
  "home",

  // Main chat application
  "apex-chat"
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/": "/apex-chat",
  "/chat": "/apex-chat"
};
