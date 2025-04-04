import { createLogger } from "@sap-cloud-sdk/util";

const logger = createLogger({
  package: "content-filtering",
  messageContext: "orchestration",
});

export { logger };
