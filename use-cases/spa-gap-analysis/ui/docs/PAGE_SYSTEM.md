# Router-Based Page System

This application uses a modern router-based page system with **page.js** for client-side routing. Each page is organized into its own directory with HTML, CSS, and JavaScript files together for better organization and maintainability.

## Directory Structure

```
src/pages/              # All page files together
├── home/
│   ├── home.html      # Page template
│   ├── home.js        # Page-specific functionality
│   └── home.css       # Page-specific styles
├── chat/
│   ├── chat.html
│   ├── chat.js
│   └── chat.css
└── [pageName]/
    ├── [pageName].html
    ├── [pageName].js
    └── [pageName].css
```

## How It Works

1. **Router**: Uses **page.js** for client-side routing with proper URL paths (e.g., `/home`, `/chat`)
2. **Dynamic Loading**: HTML, CSS, and JS are loaded dynamically when navigating to a page
3. **Code Splitting**: Vite automatically creates separate chunks for each page's JavaScript
4. **CSS Optimization**: CSS is loaded once per page and cached for performance
5. **BTP Compatible**: Works seamlessly in both development and production deployments

## Benefits of Router Approach

✅ **Modern Routing**: Uses proper URL paths instead of hash-based navigation  
✅ **Code Splitting**: Automatic JavaScript chunking for better performance  
✅ **File Organization**: All page files (HTML, CSS, JS) in same directory  
✅ **SEO Friendly**: Real URLs that can be bookmarked and shared  
✅ **Browser History**: Proper back/forward button support  
✅ **Industry Standard**: Uses established router library (page.js)  
✅ **Developer Experience**: Easier to find and maintain page-related code

## Creating a New Page

### Option 1: Using the Generator Script (Recommended)

```bash
# From the ui directory
npm run create-page <pageName> [title]

# Examples:
npm run create-page products "Product Catalog"
npm run create-page settings "User Settings"
```

**Note**: After creating a page, you'll need to:

1. ✅ Route automatically added to `src/config/routes.js`
2. Add navigation item to `index.html`

### Option 2: Manual Creation

1. Create directory: `src/pages/[pageName]/`
2. Create three files:
   - `[pageName].html` - Page template
   - `[pageName].js` - Page functionality
   - `[pageName].css` - Page styles
3. Add route to `src/config/routes.js` (just add the page name to the routes array)
4. Add navigation item to `index.html`

### HTML Template (`src/pages/[pageName]/[pageName].html`)

```html
<ui5-title>Page Title</ui5-title>
<br />
<ui5-text>Page description goes here.</ui5-text>

<div class="page-content">
  <!-- Your UI5 components and content -->
</div>
```

### JavaScript Module (`src/pages/[pageName]/[pageName].js`)

```javascript
/* Page-specific UI5 components */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
// Import other UI5 components used only on this page

/* Page-specific icons - these will load before HTML rendering */
import "@ui5/webcomponents-icons/dist/add.js";
import "@ui5/webcomponents-icons/dist/edit.js";
// Import icons used in this page's HTML template

export default function initPageName() {
  console.log("Page initialized");

  // Add event listeners, data fetching, etc.
  const button = document.getElementById("my-button");
  if (button) {
    button.addEventListener("click", () => {
      console.log("Button clicked!");
    });
  }
}
```

### CSS Styles (`src/pages/[pageName]/[pageName].css`)

```css
/* Page-specific styles */
.page-container {
  padding: 1rem;
  max-width: 1200px;
  margin: 0 auto;
}

.page-content {
  margin-top: 1rem;
}
```

## Route Configuration

Routes are now configured in a simple configuration file at `src/config/routes.js`. This eliminates the need to manually edit router code for each new page.

### Simple Route Configuration

```javascript
// src/config/routes.js
export const routes = [
  "home", // Creates route: /home
  "chat", // Creates route: /chat
  "products" // Creates route: /products
  // Add more routes here...
];
```

### Advanced Route Configuration

```javascript
// src/config/routes.js
export const routes = [
  "home",
  "chat",
  // Advanced route with custom path
  {
    path: "/admin/dashboard",
    page: "admin-dashboard",
    title: "Admin Dashboard"
  }
];
```

### Route Aliases

Create shortcuts or redirects to existing pages:

```javascript
// src/config/routes.js
export const aliases = {
  "/dashboard": "/home",
  "/ai": "/chat",
  "/admin": "/admin-dashboard"
};
```

## Adding Navigation

Add a navigation item to `index.html`:

```html
<ui5-side-navigation-item text="Page Title" href="/pageName" icon="your-icon"></ui5-side-navigation-item>
```

## Router Configuration

The router automatically reads configuration from `src/config/routes.js` and registers all routes on startup. No manual router code editing required!

```javascript
// src/config/routes.js - All you need to maintain
export const routes = [
  "home", // Automatically creates /home route
  "chat", // Automatically creates /chat route
  "about" // Automatically creates /about route
];

// Optional: Route aliases for redirects
export const aliases = {
  "/dashboard": "/home",
  "/help": "/about"
};
```

### Developer Benefits

- ✅ **No code editing**: Just add page names to the routes array
- ✅ **Auto-discovery**: Page generator automatically adds routes
- ✅ **Type safety**: Configuration is in a single, easy-to-maintain file
- ✅ **Aliases support**: Create multiple paths to the same page
- ✅ **Advanced routes**: Support for custom path mapping when needed

## File Naming Conventions

- Use lowercase, hyphenated names for multi-word pages (e.g., `user-profile`)
- Keep names concise but descriptive
- File names should match the directory name

## Best Practices

1. **CSS Scoping**: Use page-specific class prefixes to avoid conflicts
2. **JavaScript Modules**: Always export a default initialization function
3. **UI5 Component Imports**: Import UI5 components at the page level when they're only used on that page
4. **Icon Imports**: Import UI5 icons at the page level - they will be registered before HTML rendering
5. **Error Handling**: Pages should gracefully handle missing elements
6. **Responsive Design**: Include mobile-friendly styles in your CSS
7. **Accessibility**: Follow UI5 accessibility guidelines in your templates

## UI5 Component Import Strategy

To optimize bundle size and improve code organization:

- **Global components** (used in layout): Import in `main.js`
  - Shell bar, navigation, core layout components
- **Page-specific components**: Import in the respective page's JS file
  - Chat components → `chat/chat.js`
  - Card components → `home/home.js`
  - Form components → Import only in pages that use forms

### Example Import Organization

```javascript
/* main.js - Global layout components */
import "@ui5/webcomponents-fiori/dist/ShellBar.js";
import "@ui5/webcomponents-fiori/dist/SideNavigation.js";

/* pages/chat/chat.js - Chat-specific components */
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/TextArea.js";

/* pages/forms/forms.js - Form-specific components */
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/DatePicker.js";
```

## Icon Loading Solution

### The Problem

UI5 icons need to be registered **before** the HTML template is parsed and rendered. In the previous version, icons imported at the page level would not display because they were loaded after the HTML was already rendered.

### The Solution

The router now uses a **two-phase loading system**:

1. **Phase 1 (Registration)**: Import the JavaScript module to register all icons and components
2. **Phase 2 (HTML Loading)**: Load and render the HTML template (icons are now available)
3. **Phase 3 (Initialization)**: Execute the page's initialization function for DOM manipulation

### Usage

Simply import icons at the top of your page's JavaScript file:

```javascript
/* Page-specific icons - these will load before HTML rendering */
import "@ui5/webcomponents-icons/dist/add.js";
import "@ui5/webcomponents-icons/dist/edit.js";
import "@ui5/webcomponents-icons/dist/delete.js";

export default function initMyPage() {
  // Your page logic here
}
```

### Benefits

- ✅ **Icons work correctly** when imported at page level
- ✅ **Code splitting maintained** - icons only loaded when page is accessed
- ✅ **No manual router editing** - system handles the loading order automatically
- ✅ **Backward compatible** - existing pages continue to work

## Loading States & FOUC Prevention

### The Problem

Users can experience a "Flash of Unstyled Content" (FOUC) where HTML appears briefly without CSS styling, causing large images and unstyled elements to flash on screen. This happens both on initial page load and when navigating between pages.

### The Solution

The application implements a comprehensive loading system that prevents FOUC at multiple levels:

#### 1. **Application-Level FOUC Prevention**

- **HTML-level hiding**: The entire app is hidden with `opacity: 0` in inline CSS
- **Clean loading**: No visual indicators, just a clean blank page while loading
- **UI5 component registration**: App only appears after all UI5 components are loaded and registered
- **Smooth reveal**: 400ms fade-in transition when everything is ready

#### 2. **Page-Level FOUC Prevention**

- **Pre-HTML loading**: Icons and components registered before HTML parsing
- **Smooth page transitions**: Content fades out/in when navigating between pages
- **Style settling**: Brief pause ensures CSS is fully applied before content appears

### Benefits

- ✅ **Zero visual flashes** - Content only appears when fully styled
- ✅ **Professional UX** - Clean, smooth transitions without loading clutter
- ✅ **Minimal approach** - Simple blank page while loading, no distracting spinners
- ✅ **Fully automatic** - No additional code needed in page files
- ✅ **Error handling** - App appears even if loading encounters issues

### Loading Sequence

#### Initial Application Load:

```
1. HTML loads → App hidden, blank page visible
2. JavaScript starts → UI5 components begin loading
3. Router initializes → Page system ready
4. UI5 components registered → All icons and components available
5. App fades in (400ms) → App becomes visible with smooth transition
```

#### Page Navigation:

```
1. User clicks navigation → Current content fades out
2. Page JS imports → Icons/components registered
3. CSS loads → Page styles ready
4. HTML loads → Page content ready
5. Content fades in → Smooth transition complete
```

This eliminates the jarring experience of seeing unstyled content and provides a polished, professional feel throughout the entire application lifecycle.

## BTP Deployment Compatibility

The router-based system ensures full compatibility with SAP BTP:

- **Static assets**: HTML and CSS are properly bundled by Vite
- **JavaScript modules**: Automatically code-split for optimal loading
- **Development vs Production**: Same behavior in both environments
- **No additional configuration**: Works with existing BTP deployment process

## Migration from Hash-Based System

The new router system provides significant improvements over hash-based navigation:

- **Real URLs**: Proper paths instead of `#page` fragments
- **Better SEO**: Search engines can properly index pages
- **Code Splitting**: Automatic JavaScript chunking
- **Modern Standards**: Industry-standard routing approach
- **Better UX**: Cleaner URLs that can be bookmarked

## Performance Benefits

- **Code Splitting**: Each page's JavaScript is loaded only when needed
- **CSS Optimization**: Styles are loaded once per page and cached
- **Lazy Loading**: Pages are loaded on-demand, reducing initial bundle size
- **Automatic Chunking**: Vite optimizes bundle splitting automatically
- **FOUC Prevention**: Loading states prevent flash of unstyled content
- **Smooth Transitions**: Fade-in animations provide polished user experience

## Troubleshooting

**Page not loading?**

- Check that all three files exist in `src/pages/[pageName]/`
- Verify the route is added to `src/modules/router.js`
- Check browser console for JavaScript errors
- Ensure navigation href uses path format (`/pageName`)

**Styles not applying?**

- Ensure CSS file exists in the page directory
- Check for CSS class name conflicts
- Verify CSS is being imported by checking Network tab in dev tools

**JavaScript not working?**

- Confirm the JS file exports a default function
- Check for JavaScript errors in browser console
- Ensure DOM elements exist before attaching listeners
- Verify module imports are correct

**Routing issues?**

- Check that page.js is installed (`npm list page`)
- Verify route is properly defined in router.js
- Check that navigation uses `/path` format, not `#hash`
- Ensure router.start() is called

**Build/deployment issues?**

- Verify all imports use proper ES module syntax
- Check that Vite can resolve all page modules
- Ensure no circular dependencies between modules
- Test build with `npm run build` before deployment
