import dotenv from "dotenv";
dotenv.config();
import type { OrchestrationResponse } from "@sap-ai-sdk/orchestration";
import { OrchestrationClient } from "@sap-ai-sdk/orchestration";

/**
 * A simple LLM request, asking about the capital of France.
 * @returns The orchestration service response.
 */
export async function askCapitalOfCountry(
  country: string
): Promise<OrchestrationResponse> {
  const orchestrationClient = new OrchestrationClient({
    llm: {
      model_name: "gpt-4-32k",
    },
    templating: {
      template: [
        {
          role: "user",
          content: "What is the capital of {{?country}}?",
        },
      ],
    },
  });

  const response = await orchestrationClient.chatCompletion({
    inputParams: { country: country },
  });
  return response;
}
