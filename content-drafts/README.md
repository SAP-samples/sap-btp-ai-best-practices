# Content Drafts

Raw markdown drafts exported from SharePoint pages, before they're edited and moved into `docs/`.

## Workflow: SharePoint → Markdown

1. Open the SharePoint page in Chrome or Edge
2. Open DevTools console (`F12` → Console tab)
3. Paste the contents of [`sharepoint-to-md-converter.js`](sharepoint-to-md-converter.js) and press Enter
4. Run:
   ```js
   await copySharePointPageAsMarkdown()
   ```
5. The markdown is copied to your clipboard (or a fallback dialog appears)
6. Paste into a new `.md` file in this folder, named after the page

### Options

```js
// Skip images (useful when they're all blob/data URLs anyway)
await copySharePointPageAsMarkdown({ includeImages: false })

// Add YAML front matter with title, source URL, and capture date
await copySharePointPageAsMarkdown({ includeFrontMatter: true })

// Convert only a specific section
await copySharePointPageAsMarkdown({ rootSelector: '#my-section' })

// Convert your current text selection instead of the full page
await copySharePointPageAsMarkdown({ preferSelection: true })
```

### What happens to unexportable images?

SharePoint often renders images as `blob:` or `data:` URLs that can't be saved via copy-paste. By default the converter inserts an italic placeholder like:

> _[Image not exported from SharePoint (blob URL)]_

You can change this with `unexportableImageHandling: "skip"` to silently drop them.

## After export

- Review and clean up the markdown
- Move the file into the appropriate `docs/` subdirectory
- Replace image placeholders with actual assets (download from SharePoint and place in `docs/assets/`)
