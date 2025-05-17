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
    const { ScienceDataUpload } = this.entities("RagService");

    // Hooks
    this.on("PUT", ScienceDataUpload, this.onImportScienceData);

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
}
