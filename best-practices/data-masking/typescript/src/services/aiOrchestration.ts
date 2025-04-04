import dotenv from "dotenv";
dotenv.config();
import type { OrchestrationResponse } from "@sap-ai-sdk/orchestration";
import { OrchestrationClient } from "@sap-ai-sdk/orchestration";

/**
 * A simple LLM request, asking about the capital of France.
 * @returns The orchestration service response.
 */
export async function orchestrationChatCompletionMasking(
  prompt: string,
  anonymize: boolean
): Promise<OrchestrationResponse> {
  const orchestrationClient = new OrchestrationClient({
    llm: {
      model_name: "gpt-4-32k",
    },
    templating: {
      template: [
        {
          role: "user",
          content: prompt,
        },
      ],
    },
    masking: {
      masking_providers: [
        {
          type: "sap_data_privacy_integration",
          method: anonymize ? "anonymization" : "pseudonymization",
          entities: [{ type: "profile-email" }, { type: "profile-person" }],
        },
      ],
    },
  });

  const response = await orchestrationClient.chatCompletion();
  return response;
}
