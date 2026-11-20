# ECN Impact page lifecycle

The UI uses `page` for same-tab navigation and loads each route as a separate HTML, CSS, and JavaScript module. Route definitions live in `src/routes.js`; `src/modules/router.js` owns the active page lifecycle.

## Page contract

Each page exports `init(context)`, either directly or as its default export while the temporary Home compatibility route exists. The initializer may be asynchronous and returns an optional cleanup function.

The context contains:

- `params`: decoded route parameters supplied by the router.
- `query`: a `URLSearchParams` instance for the current route.
- `container`: the page content element.
- `navigate(path)`: same-tab navigation through `page`.
- `signal`: an `AbortSignal` cancelled when a newer navigation begins.

The router awaits the previous cleanup before activating the next page. A navigation generation prevents a slow earlier initialization from overwriting newer content; if stale initialization eventually returns a cleanup function, that cleanup runs immediately.

## Network request lifecycle

`createLatestRequestGate(parentSignal)` creates one last-write-wins lane. Starting a request aborts only the previous request in that same lane. Pages create separate lanes for independent operations such as list reads, progress reads, and upload mutations, so a refresh cannot cancel a user action. Disposing a page aborts all of its lanes through the route signal.

## Historical-list URL state

`parseListState()` accepts only the backend-supported filter, sort, order, and pagination values. `serializeListState()` emits only that allowlist. Detail links use immutable `ecn_row_id` values and carry the list state so Back navigation restores the originating view.

## Safety and not-found behavior

The not-found page is constructed with DOM APIs and event listeners. It contains no inline handler or interpolated HTML. Source and model content follow their own text/sanitization boundaries in the product pages.

## Verification

From the repository root:

```bash
npm --prefix ui test -- tests/navigation.test.js tests/request-generation.test.js tests/list-state.test.js
npm --prefix ui run test:e2e -- shell.spec.js
npm --prefix ui run build
```
