import "dotenv/config";

export type ExampleSelector = "without-schema" | "with-schema" | "all";

export interface Rpt1Config {
  deploymentId: string;
}

export function readConfig(): Rpt1Config {
  const serviceKey = process.env.AICORE_SERVICE_KEY?.trim();
  const deploymentId = process.env.RPT1_DEPLOYMENT_ID?.trim();

  if (!serviceKey) {
    throw new Error("Missing required environment variable: AICORE_SERVICE_KEY");
  }

  if (!deploymentId) {
    throw new Error("Missing required environment variable: RPT1_DEPLOYMENT_ID");
  }

  return { deploymentId };
}

export function readSelector(argv: string[]): ExampleSelector {
  const value = argv[2]?.trim().toLowerCase() ?? "without-schema";
  if (value === "without-schema" || value === "with-schema" || value === "all") {
    return value;
  }

  throw new Error(
    `Unsupported example selector "${value}". Expected one of: without-schema, with-schema, all.`
  );
}
