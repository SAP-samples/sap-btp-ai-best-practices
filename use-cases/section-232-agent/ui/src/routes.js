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
  "products",
  "settings",
  {
    path: "/settings/section-232/review",
    page: "section-232-review"
  },
  {
    path: "/products/:itemId",
    page: "product-detail"
  }

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
  "/dashboard": "/products",
  "/home": "/products"
  // Add more aliases as needed
};
