import { createLogger } from "@sap-cloud-sdk/util";

const logger = createLogger({
  package: "data-masking",
  messageContext: "orchestration",
});

export { logger };
