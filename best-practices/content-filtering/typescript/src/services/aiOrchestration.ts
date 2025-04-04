import dotenv from "dotenv";
dotenv.config();
import type { OrchestrationResponse } from "@sap-ai-sdk/orchestration";
import {
  OrchestrationClient,
  buildAzureContentSafetyFilter,
} from "@sap-ai-sdk/orchestration";

// create a filter with minimal thresholds for hate and violence
const azureContentFilter = buildAzureContentSafetyFilter({
  Hate: "ALLOW_SAFE",
  Violence: "ALLOW_SAFE",
});
// create a filter with no thresholds for hate and violence
const azureContentNoFilter = buildAzureContentSafetyFilter({
  Hate: "ALLOW_ALL",
  Violence: "ALLOW_ALL",
});

/**
 * A simple LLM request sending input to the LLM and returning the response.
 * @returns The orchestration service response.
 */
export async function chatWithAgent(
  input: string,
  filterInput: boolean = true
): Promise<OrchestrationResponse> {
  const orchestrationClient = new OrchestrationClient({
    llm: {
      model_name: "gpt-4o",
    },
    templating: {
      template: [
        {
          role: "user",
          content: "{{?input}}",
        },
      ],
    },
    // configure the filter to be applied for input
    filtering: {
      input: {
        filters: [filterInput ? azureContentFilter : azureContentNoFilter],
      },
    },
  });

  return orchestrationClient.chatCompletion({
    inputParams: { input: input },
  });
}

/**
 * A simple LLM request generating a paraphrase and returning the response.
 * @returns The orchestration service response.
 */
export async function generateParaphrase(
  input: string,
  filterOutput: boolean = true
): Promise<OrchestrationResponse> {
  const orchestrationClient = new OrchestrationClient({
    llm: {
      model_name: "gpt-4o",
    },
    templating: {
      template: [
        {
          role: "system",
          content: 'Create 3 paraphrases of the sentence: "{{?input}}"',
        },
      ],
    },
    // only filter the output
    filtering: {
      output: {
        filters: [filterOutput ? azureContentFilter : azureContentNoFilter],
      },
    },
  });

  return orchestrationClient.chatCompletion({
    inputParams: { input: input },
  });
}
