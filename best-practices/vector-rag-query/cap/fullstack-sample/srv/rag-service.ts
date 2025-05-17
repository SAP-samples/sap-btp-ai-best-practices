import cds from "@sap/cds";
import { parseFile } from "./utils/csv-file.js";
import { IScienceData } from "./types.js";
import AiSdk from "./services/ai-sdk.js";

const { uuid } = cds.utils;
const logger = cds.log("rag");
const chunkSize = 500;
const embeddingBatchSize = 100;
const oAiSdk = new AiSdk();

export default class RagService extends cds.ApplicationService {
  async init() {
    const { ScienceDataUpload } = this.entities;
    const { queryScienceData } = this.operations;

    // Hooks
    this.on("PUT", ScienceDataUpload, this.onImportScienceData);
    this.on(queryScienceData, this.onQueryScienceData);

    // Super
    await super.init();
  }

  /**
   * Handler for importing science data from CSV file
   * @param {Request} req
   * @returns {Promise<any>}
   */
  private readonly onImportScienceData = async (req: cds.Request) => {
    try {
      const { ScienceData } = this.entities;

      if (req.data.content && req.data.mediaType === "text/csv") {
        // Process csv file
        let scienceDataRows = await parseFile(req.data.content);

        /** Loop through dataap fields to correct column names, add uuid
         *  - Map fields to correct column names
         *  - Add UUID for database ID
         *  - Break into chunks for embedding
         * **/
        let scienceDataChunks: any[] = [];
        scienceDataRows.forEach((row: any) => {
          const { "Difficulty Level": DifficultyLevel, ...rest } = row;
          const newRow: IScienceData = { ...rest, DifficultyLevel };

          // Break into chunks
          let index = 0;
          while (index < newRow.Topic.length) {
            scienceDataChunks.push({ ...newRow, ID: uuid(), Topic: newRow.Topic.slice(index, index + chunkSize) });
            index += chunkSize;
          }
        });

        // Create Embeddings
        const embeddingResults = await oAiSdk.createEmbeddings(scienceDataChunks, embeddingBatchSize);

        // Store in DB
        const result = await INSERT.into(ScienceData).entries(embeddingResults);

        req.reply(result);
      } else {
        logger.error("Invalid file type or missing content");
        req.error({ code: 1, status: 400, message: "Invalid file type or missing content" });
      }
    } catch (ex) {
      if (ex instanceof Error) {
        logger.error(`Error while uploading file: ${ex.message}`);
        req.error({ code: 1, status: 400, message: `Error while uploading file: ${ex.message}` });
      }
    }
  };

  /**
   * Handler for query science data
   * @param {Request} req
   * @returns {Promise<any>}
   */
  private readonly onQueryScienceData = async (req: cds.Request) => {
    try {
      // Check Parameters
      const { query } = req.data;
      const top = 5; //Number of vector documents to include

      if (query) {
        // Get Embedding for query
        const queryVector = await oAiSdk.createEmbedding(query);

        // Query the database
        const vectorResult = await cds.run(
          `SELECT TOP ${top} ID,TOPIC,DIFFICULTYLEVEL,CATEGORY FROM AI_DB_SCIENCE_DATA ORDER BY L2DISTANCE(EMBEDDING, TO_REAL_VECTOR('${queryVector}')) ASC;`
        );
        const context = vectorResult.reduce((outputString: string, row: any) => {
          return (outputString += `${row.TOPIC} `);
        }, "");

        // Create prompt
        const prompt = `Use the following context information to answer to user's query.
          Here is some context: ${context}

          Based on the above context, answer the following query:
          ${query}

          The answer tone has to be very professional in nature.

          If you don't know the answer, politely say that you don't know, don't try to make up an answer.
          `;

        // Ask the LLM
        const response = await oAiSdk.chatCompletion(prompt);
        req.reply(response.getContent(0));
      } else {
        req.error(400, "Parameter 'query' is required.");
      }
    } catch (ex) {
      if (ex instanceof Error) {
        logger.error(`Error while uploading file: ${ex.message}`);
        req.error({ code: 1, status: 400, message: `Error while uploading file: ${ex.message}` });
      }
    }
  };
}
