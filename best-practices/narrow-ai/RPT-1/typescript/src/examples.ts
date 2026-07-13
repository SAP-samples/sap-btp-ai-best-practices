import { RptClient } from "@sap-ai-sdk/rpt";
import type { Rpt1Config } from "./config.js";
import {
  classificationPredictionData,
  discountDataSchema,
  discountPredictionData,
} from "./data.js";

export async function runWithoutSchema(config: Rpt1Config) {
  const rptClient = new RptClient({ deploymentId: config.deploymentId });
  return rptClient.predictWithoutSchema(classificationPredictionData);
}

export async function runWithSchema(config: Rpt1Config) {
  const rptClient = new RptClient({ deploymentId: config.deploymentId });
  return rptClient.predictWithSchema(discountDataSchema, discountPredictionData);
}
