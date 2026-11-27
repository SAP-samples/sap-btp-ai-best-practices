/**
 * Browser regression for profile/dataset loading after configuration help mounts.
 * Run these commands from the repository root:
 *   npm --prefix ui run dev -- --host 127.0.0.1 --port 5187 --strictPort
 *   npx @playwright/cli -s=configuration-help open http://127.0.0.1:5187
 *   npx @playwright/cli -s=configuration-help run-code "$(cat ui/tests/configuration_help.browser.js)"
 * Input: a Playwright page. Output: a PASS message, or a thrown regression error.
 * API requests use disposable fixtures; no backend or HANA connection is needed.
 */
async (page) => {
  const origin = "http://127.0.0.1:5187";
  const settings = { matrix_mode: "FLEXIBLE", maximum_group_size: 7 };
  const profiles = [
    { profile_id: "profile-a", name: "Profile A", plant: "P1", revision: 2, settings },
    { profile_id: "profile-b", name: "Profile B", plant: "P1", revision: 1, settings: { ...settings, matrix_mode: "HARD" } },
  ];
  const datasets = [
    { dataset_id: "dataset-a", name: "Dataset A", status: "published", metadata: { plant: "P1" } },
    { dataset_id: "dataset-b", name: "Dataset B", status: "published", metadata: { plant: "P1" } },
  ];
  const guide = {
    "workspace-dataset": { title: "Published dataset", text: "Select the data for this run." },
    matrix: { title: "Compatibility matrix", text: "The scenario may change the profile default.", fields: { HARD: "Strict compatibility." } },
  };
  const drafts = new Map();
  const errors = [];
  const calls = [];
  await page.unroute("**/api/**");
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    calls.push({ path, method });
    let data = {};
    if (path === "/api/configuration-help") data = guide;
    else if (path === "/api/plant-profiles") data = { profiles };
    else if (path.startsWith("/api/plant-profiles/")) data = profiles.find((profile) => path.endsWith(profile.profile_id));
    else if (path === "/api/datasets") data = { items: datasets };
    else if (path.startsWith("/api/datasets/")) data = datasets.find((dataset) => path.endsWith(dataset.dataset_id));
    else if (path === "/api/run-drafts" && method === "POST") {
      const body = request.postDataJSON();
      const profile = profiles.find((entry) => entry.profile_id === body.plant_profile_id);
      data = {
        draft_id: `draft-${drafts.size + 1}`, dataset_id: body.dataset_id,
        plant_profile_id: profile.profile_id, plant_profile: profile,
        revision: 1, title: body.title, budget: {},
        request: { config: { matrix_mode: profile.settings.matrix_mode }, scope: [], constraints: [] },
      };
      drafts.set(data.draft_id, data);
    } else if (path.startsWith("/api/run-drafts/")) {
      const draft = drafts.get(path.split("/")[3]);
      if (method === "PATCH") {
        const patch = request.postDataJSON();
        Object.assign(draft, patch, { revision: draft.revision + 1 });
      }
      data = draft;
    } else if (path === "/api/runs") data = { items: [] };
    else if (path === "/api/query") data = { rows: [], fields: [], total: 0 };
    else throw new Error(`Unexpected fixture request: ${method} ${path}`);
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(data) });
  });
  // Start with a fresh selection even if the CLI browser was already open.
  await page.goto(origin);
  await page.evaluate(() => localStorage.removeItem("production-wheel.selection"));
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${origin}/#workspace`);
  await page.getByRole("button", { name: "About Compatibility matrix", exact: true }).waitFor();
  await page.locator("#workspace-profile").selectOption("profile-a");
  await page.locator("#workspace-dataset").selectOption("dataset-a");
  // A successful POST must be followed by a complete draft render, not an error banner.
  await page.waitForFunction(() =>
    !document.querySelector("#notice").hidden
    || document.querySelector("#draft-revision").textContent === "Draft · revision 1",
  );
  if (await page.locator("#notice").isVisible()) throw new Error(await page.locator("#notice").textContent());
  if (await page.locator("#matrix-default").textContent() !== "Plant profile default: Avoid allowed")
    throw new Error("The plant profile default did not survive help mounting.");
  await page.locator("#run-list").getByText("No runs for this snapshot yet.", { exact: true }).waitFor();
  const savedNode = await page.locator("#matrix-default").elementHandle();

  // The help dialog still opens and closes after draft rendering.
  await page.getByRole("button", { name: "About Compatibility matrix", exact: true }).click();
  await page.locator("dialog[open]").getByText(guide.matrix.text, { exact: true }).waitFor();
  await page.locator("dialog[open]").getByRole("button", { name: "Close", exact: true }).click();

  // Scenario overrides and repeated dataset/profile changes reuse the preserved node.
  await page.locator("#matrix").selectOption("DIAGNOSTIC");
  await page.locator("#save-draft").click();
  await page.waitForFunction(() => document.querySelector("#draft-revision").textContent === "Draft · revision 2");
  if (drafts.get("draft-1").request.config.matrix_mode !== "DIAGNOSTIC")
    throw new Error("Saving the scenario did not retain its compatibility override.");
  await page.locator("#workspace-dataset").selectOption("dataset-b");
  await page.waitForFunction(() => document.querySelector("#draft-revision").textContent === "Draft · revision 1");
  await page.locator("#workspace-profile").selectOption("profile-b");
  await page.waitForFunction(() => document.querySelector("#matrix-default").textContent === "Plant profile default: Strict");
  if (!await savedNode.evaluate((node) => node === document.querySelector("#matrix-default")))
    throw new Error("Help or draft rendering replaced the supporting element.");

  // Returning to the workspace exercises GET restoration with help enabled.
  await page.locator("[data-nav=datasets]").click();
  await page.locator("#review-title").waitFor();
  await page.locator("[data-nav=workspace]").click();
  await page.getByRole("button", { name: "About Compatibility matrix", exact: true }).waitFor();
  await page.waitForFunction(() => document.querySelector("#matrix-default")?.textContent === "Plant profile default: Strict");
  if (!calls.some((call) => call.path === "/api/run-drafts/draft-3" && call.method === "GET"))
    throw new Error("The selected draft was not restored from the API.");
  if (await page.locator("#notice").isVisible()) throw new Error(await page.locator("#notice").textContent());

  // Exercise richer labels and section headings with the same shared help builder.
  await page.route("**/api/help-fixture", (route) => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({
      fixture: { title: "Checkbox setting", text: "A label after its input." },
      section: { title: "Settings section", text: "A standalone heading." },
    }),
  }));
  await page.evaluate(async () => {
    const { mountConfigurationHelp } = await import("/src/workspace/help.js");
    const target = document.createElement("section");
    target.innerHTML = '<div class="configuration"><label><input id="fixture" type="checkbox"> Only mixed groups<span id="support">Supporting text</span><a href="#support">Details</a></label><h2 id="section">Section heading</h2></div>';
    document.body.append(target);
    const control = target.querySelector("#fixture");
    const support = target.querySelector("#support");
    let clicked = false;
    support.addEventListener("click", () => { clicked = true; });
    await mountConfigurationHelp(target, "/api/help-fixture");
    if (target.querySelector("#fixture") !== control || target.querySelector("#support") !== support)
      throw new Error("The help builder dropped or cloned a label's child elements.");
    const label = target.querySelector('label[for="fixture"]');
    if (label.textContent !== "Only mixed groups" || !control.labels.length)
      throw new Error("The checkbox lost its accessible label.");
    support.click();
    if (!clicked || !target.querySelector('a[href="#support"]'))
      throw new Error("The supporting content or its event handlers were lost.");
    if (!target.querySelector("#section .info-button"))
      throw new Error("The section heading lost its inline help button.");
    target.remove();
  });
  if (errors.length) throw new Error(errors.join("\n"));
  console.log("PASS: profile/dataset selection, saving overrides, profile switching, draft restoration, help dialogs, supporting elements, checkbox labels and section headings; no browser errors.");
}
