import { readConfig, readSelector } from "./config.js";
import { runWithSchema, runWithoutSchema } from "./examples.js";
import { printResult } from "./output.js";

async function main(): Promise<void> {
  const config = readConfig();
  const selector = readSelector(process.argv);

  if (selector === "without-schema" || selector === "all") {
    const response = await runWithoutSchema(config);
    printResult("without-schema", response);
  }

  if (selector === "with-schema" || selector === "all") {
    const response = await runWithSchema(config);
    printResult("with-schema", response);
  }
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
