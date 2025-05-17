import {
  AzureOpenAiChatClient,
  AzureOpenAiChatCompletionResponse,
  AzureOpenAiEmbeddingClient,
} from "@sap-ai-sdk/foundation-models";
import { IScienceData } from "../types";

export default class AiSdk {
  public resourceGroupId: string = "default";

  /**
   * Create Embeddings
   * @param {Array<IScienceData>} science data - An array of science data.
   * @return {Promise} - Returns a Promise that resolves to an array of embeddings.
   */
  public createEmbeddings = async (rows: Array<IScienceData>, batchSize: number): Promise<any> => {
    try {
      const client = this.getEmbeddingClient(this.resourceGroupId);

      for (let i = 0; i < rows.length / batchSize; i++) {
        const startIndex = i * batchSize;
        const batchTopics = rows.slice(startIndex, startIndex + batchSize).map((row) => row.Topic);
        const batchResponse = await client.run({ input: batchTopics });
        const batchEmbeddings = batchResponse.getEmbeddings();
        batchEmbeddings.forEach((response: any, idx: number) => {
          rows[startIndex + idx].Embedding = `[${response!.join(",")}]`;
        });
      }
      return rows;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Error creating embeddings";
      throw new Error(errorMessage);
    }
  };

  public createEmbedding = async (query: string): Promise<any> => {
    try {
      const client = this.getEmbeddingClient(this.resourceGroupId);
      const embeddingResponse = await client.run({ input: query });
      return `[${embeddingResponse.getEmbedding(0)}]`;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Error creating embeddings";
      throw new Error(errorMessage);
    }
  };

  public chatCompletion = async (prompt: string): Promise<AzureOpenAiChatCompletionResponse> => {
    try {
      const client = this.getChatCompletionClient(this.resourceGroupId);
      const chatResponse = await client.run({
        messages: [
          { role: "system", content: "You are an intelligent assistant." },
          { role: "user", content: prompt },
        ],
      });
      return chatResponse;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Error in chat request.";
      throw new Error(errorMessage);
    }
  };

  private readonly getEmbeddingClient = (resourceGroupId: string) => {
    return new AzureOpenAiEmbeddingClient({
      modelName: "text-embedding-3-small",
    });
  };

  private readonly getChatCompletionClient = (resourceGroupId: string) => {
    return new AzureOpenAiChatClient({
      modelName: "gpt-4o",
    });
  };
}
