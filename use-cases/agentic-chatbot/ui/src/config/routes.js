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
  // Home page (mapped to root and /home)
  "home",

  // Main application pages
  "chat-with-tools"
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/chat": "/chat-with-tools"
};
