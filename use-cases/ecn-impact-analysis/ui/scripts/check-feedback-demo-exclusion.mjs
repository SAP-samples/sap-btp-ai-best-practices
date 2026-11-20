// Verify actual build output: a DEV guard alone cannot exclude modules caught by a broad importer.
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { resolve, join, relative } from "node:path";

const root = resolve(process.argv[2] || "dist");
const marker = /linked-ecn-demo|demo-linked-ecn|ECN-DEMO-|Simulated demo relationship/;
const findings = [];
let scanned = 0;
async function scan(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) { await scan(path); continue; }
    scanned++;
    if (marker.test(item.name) || (/\.(?:js|css|html|json|map)$/.test(item.name) && marker.test(await readFile(path, "utf8")))) {
      findings.push(relative(root, path));
    }
  }
}
await scan(root);
assert.ok(scanned > 0, "Build files are required; run npm run build first.");
assert.deepEqual(findings, [], "DEV Linked ECN demo was included in production build");
console.log(`PASS: ${scanned} production files checked; no Linked ECN demo assets or literals.`);
