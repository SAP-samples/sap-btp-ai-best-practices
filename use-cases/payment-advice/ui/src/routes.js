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
  // Screen 1: Payment Advice Extractor (mapped to root and /home)
  "home",

  // Screen 2: Deduction Rules Assistant chat
  "rules"
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/dashboard": "/home"
  // Add more aliases as needed
};
