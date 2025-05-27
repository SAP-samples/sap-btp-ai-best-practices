import { createLogger } from "@sap-cloud-sdk/util";

const logger = createLogger({
  package: "prompt-registry",
  messageContext: "orchestration",
});

export { logger };
