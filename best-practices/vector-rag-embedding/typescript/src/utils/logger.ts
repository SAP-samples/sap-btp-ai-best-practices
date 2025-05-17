import { createLogger } from "@sap-cloud-sdk/util";

const logger = createLogger({
  package: "rag-embedding",
  messageContext: "aisdk",
});

export { logger };
