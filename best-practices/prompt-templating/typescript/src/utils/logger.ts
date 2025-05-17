import { createLogger } from "@sap-cloud-sdk/util";

const logger = createLogger({
  package: "prompt-templating",
  messageContext: "orchestration",
});

export { logger };
