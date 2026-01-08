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
  "chat",
  "chat-with-history",
  "chat-with-tools",
  "chat-with-history-and-grounding",
  "document-intelligence",
  "utilities-rate-compare-and-export",
  "prompts-management"

  // Example of advanced route configuration:
  // {
  //   path: '/custom-path',
  //   page: 'my-page',
  //   title: 'Custom Page Title'
  // }
];

/**
 * Route aliases - redirects one path to another
 */
export const aliases = {
  "/dashboard": "/home"
  // Add more aliases as needed
};
