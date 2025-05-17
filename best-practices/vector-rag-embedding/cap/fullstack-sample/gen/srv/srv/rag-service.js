import cds from "@sap/cds";
import { parseFile } from "./utils/csv-file.js";
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
    onImportScienceData = async (req) => {
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
                let scienceDataChunks = [];
                scienceDataRows.forEach((row) => {
                    const { "Difficulty Level": DifficultyLevel, ...rest } = row;
                    const newRow = { ...rest, DifficultyLevel };
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
            }
            else {
                logger.error("Invalid file type or missing content");
                req.error({ code: 1, status: 400, message: "Invalid file type or missing content" });
            }
        }
        catch (ex) {
            if (ex instanceof Error) {
                logger.error(`Error while uploading file: ${ex.message}`);
                req.error({ code: 1, status: 400, message: `Error while uploading file: ${ex.message}` });
            }
        }
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFnLXNlcnZpY2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi9zcnYvcmFnLXNlcnZpY2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUEsT0FBTyxHQUFHLE1BQU0sVUFBVSxDQUFDO0FBQzNCLE9BQU8sRUFBRSxTQUFTLEVBQUUsTUFBTSxxQkFBcUIsQ0FBQztBQUVoRCxPQUFPLEtBQUssTUFBTSxzQkFBc0IsQ0FBQztBQUV6QyxNQUFNLEVBQUUsSUFBSSxFQUFFLEdBQUcsR0FBRyxDQUFDLEtBQUssQ0FBQztBQUMzQixNQUFNLE1BQU0sR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQzlCLE1BQU0sU0FBUyxHQUFHLEdBQUcsQ0FBQztBQUN0QixNQUFNLGtCQUFrQixHQUFHLEdBQUcsQ0FBQztBQUMvQixNQUFNLE1BQU0sR0FBRyxJQUFJLEtBQUssRUFBRSxDQUFDO0FBRTNCLE1BQU0sQ0FBQyxPQUFPLE9BQU8sVUFBVyxTQUFRLEdBQUcsQ0FBQyxrQkFBa0I7SUFDNUQsS0FBSyxDQUFDLElBQUk7UUFDUixNQUFNLEVBQUUsaUJBQWlCLEVBQUUsR0FBRyxJQUFJLENBQUMsUUFBUSxDQUFDLFlBQVksQ0FBQyxDQUFDO1FBRTFELFFBQVE7UUFDUixJQUFJLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxpQkFBaUIsRUFBRSxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUU1RCxNQUFNLEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUNyQixDQUFDO0lBRUQ7Ozs7T0FJRztJQUNjLG1CQUFtQixHQUFHLEtBQUssRUFBRSxHQUFnQixFQUFFLEVBQUU7UUFDaEUsSUFBSSxDQUFDO1lBQ0gsTUFBTSxFQUFFLFdBQVcsRUFBRSxHQUFHLElBQUksQ0FBQyxRQUFRLENBQUM7WUFFdEMsSUFBSSxHQUFHLENBQUMsSUFBSSxDQUFDLE9BQU8sSUFBSSxHQUFHLENBQUMsSUFBSSxDQUFDLFNBQVMsS0FBSyxVQUFVLEVBQUUsQ0FBQztnQkFDMUQsbUJBQW1CO2dCQUNuQixJQUFJLGVBQWUsR0FBRyxNQUFNLFNBQVMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO2dCQUV4RDs7OztzQkFJTTtnQkFDTixJQUFJLGlCQUFpQixHQUFVLEVBQUUsQ0FBQztnQkFDbEMsZUFBZSxDQUFDLE9BQU8sQ0FBQyxDQUFDLEdBQVEsRUFBRSxFQUFFO29CQUNuQyxNQUFNLEVBQUUsa0JBQWtCLEVBQUUsZUFBZSxFQUFFLEdBQUcsSUFBSSxFQUFFLEdBQUcsR0FBRyxDQUFDO29CQUM3RCxNQUFNLE1BQU0sR0FBaUIsRUFBRSxHQUFHLElBQUksRUFBRSxlQUFlLEVBQUUsQ0FBQztvQkFFMUQsb0JBQW9CO29CQUNwQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7b0JBQ2QsT0FBTyxLQUFLLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxNQUFNLEVBQUUsQ0FBQzt3QkFDbkMsaUJBQWlCLENBQUMsSUFBSSxDQUFDLEVBQUUsR0FBRyxNQUFNLEVBQUUsRUFBRSxFQUFFLElBQUksRUFBRSxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxLQUFLLEVBQUUsS0FBSyxHQUFHLFNBQVMsQ0FBQyxFQUFFLENBQUMsQ0FBQzt3QkFDdkcsS0FBSyxJQUFJLFNBQVMsQ0FBQztvQkFDckIsQ0FBQztnQkFDSCxDQUFDLENBQUMsQ0FBQztnQkFFSCxvQkFBb0I7Z0JBQ3BCLE1BQU0sZ0JBQWdCLEdBQUcsTUFBTSxNQUFNLENBQUMsZ0JBQWdCLENBQUMsaUJBQWlCLEVBQUUsa0JBQWtCLENBQUMsQ0FBQztnQkFFOUYsY0FBYztnQkFDZCxNQUFNLE1BQU0sR0FBRyxNQUFNLE1BQU0sQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsT0FBTyxDQUFDLGdCQUFnQixDQUFDLENBQUM7Z0JBRXhFLEdBQUcsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDcEIsQ0FBQztpQkFBTSxDQUFDO2dCQUNOLE1BQU0sQ0FBQyxLQUFLLENBQUMsc0NBQXNDLENBQUMsQ0FBQztnQkFDckQsR0FBRyxDQUFDLEtBQUssQ0FBQyxFQUFFLElBQUksRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxPQUFPLEVBQUUsc0NBQXNDLEVBQUUsQ0FBQyxDQUFDO1lBQ3ZGLENBQUM7UUFDSCxDQUFDO1FBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQztZQUNaLElBQUksRUFBRSxZQUFZLEtBQUssRUFBRSxDQUFDO2dCQUN4QixNQUFNLENBQUMsS0FBSyxDQUFDLCtCQUErQixFQUFFLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQztnQkFDMUQsR0FBRyxDQUFDLEtBQUssQ0FBQyxFQUFFLElBQUksRUFBRSxDQUFDLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxPQUFPLEVBQUUsK0JBQStCLEVBQUUsQ0FBQyxPQUFPLEVBQUUsRUFBRSxDQUFDLENBQUM7WUFDNUYsQ0FBQztRQUNILENBQUM7SUFDSCxDQUFDLENBQUM7Q0FDSCIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBjZHMgZnJvbSBcIkBzYXAvY2RzXCI7XG5pbXBvcnQgeyBwYXJzZUZpbGUgfSBmcm9tIFwiLi91dGlscy9jc3YtZmlsZS5qc1wiO1xuaW1wb3J0IHsgSVNjaWVuY2VEYXRhIH0gZnJvbSBcIi4vdHlwZXMuanNcIjtcbmltcG9ydCBBaVNkayBmcm9tIFwiLi9zZXJ2aWNlcy9haS1zZGsuanNcIjtcblxuY29uc3QgeyB1dWlkIH0gPSBjZHMudXRpbHM7XG5jb25zdCBsb2dnZXIgPSBjZHMubG9nKFwicmFnXCIpO1xuY29uc3QgY2h1bmtTaXplID0gNTAwO1xuY29uc3QgZW1iZWRkaW5nQmF0Y2hTaXplID0gMTAwO1xuY29uc3Qgb0FpU2RrID0gbmV3IEFpU2RrKCk7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIFJhZ1NlcnZpY2UgZXh0ZW5kcyBjZHMuQXBwbGljYXRpb25TZXJ2aWNlIHtcbiAgYXN5bmMgaW5pdCgpIHtcbiAgICBjb25zdCB7IFNjaWVuY2VEYXRhVXBsb2FkIH0gPSB0aGlzLmVudGl0aWVzKFwiUmFnU2VydmljZVwiKTtcblxuICAgIC8vIEhvb2tzXG4gICAgdGhpcy5vbihcIlBVVFwiLCBTY2llbmNlRGF0YVVwbG9hZCwgdGhpcy5vbkltcG9ydFNjaWVuY2VEYXRhKTtcblxuICAgIGF3YWl0IHN1cGVyLmluaXQoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBIYW5kbGVyIGZvciBpbXBvcnRpbmcgc2NpZW5jZSBkYXRhIGZyb20gQ1NWIGZpbGVcbiAgICogQHBhcmFtIHtSZXF1ZXN0fSByZXFcbiAgICogQHJldHVybnMge1Byb21pc2U8YW55Pn1cbiAgICovXG4gIHByaXZhdGUgcmVhZG9ubHkgb25JbXBvcnRTY2llbmNlRGF0YSA9IGFzeW5jIChyZXE6IGNkcy5SZXF1ZXN0KSA9PiB7XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHsgU2NpZW5jZURhdGEgfSA9IHRoaXMuZW50aXRpZXM7XG5cbiAgICAgIGlmIChyZXEuZGF0YS5jb250ZW50ICYmIHJlcS5kYXRhLm1lZGlhVHlwZSA9PT0gXCJ0ZXh0L2NzdlwiKSB7XG4gICAgICAgIC8vIFByb2Nlc3MgY3N2IGZpbGVcbiAgICAgICAgbGV0IHNjaWVuY2VEYXRhUm93cyA9IGF3YWl0IHBhcnNlRmlsZShyZXEuZGF0YS5jb250ZW50KTtcblxuICAgICAgICAvKiogTG9vcCB0aHJvdWdoIGRhdGFhcCBmaWVsZHMgdG8gY29ycmVjdCBjb2x1bW4gbmFtZXMsIGFkZCB1dWlkXG4gICAgICAgICAqICAtIE1hcCBmaWVsZHMgdG8gY29ycmVjdCBjb2x1bW4gbmFtZXNcbiAgICAgICAgICogIC0gQWRkIFVVSUQgZm9yIGRhdGFiYXNlIElEXG4gICAgICAgICAqICAtIEJyZWFrIGludG8gY2h1bmtzIGZvciBlbWJlZGRpbmdcbiAgICAgICAgICogKiovXG4gICAgICAgIGxldCBzY2llbmNlRGF0YUNodW5rczogYW55W10gPSBbXTtcbiAgICAgICAgc2NpZW5jZURhdGFSb3dzLmZvckVhY2goKHJvdzogYW55KSA9PiB7XG4gICAgICAgICAgY29uc3QgeyBcIkRpZmZpY3VsdHkgTGV2ZWxcIjogRGlmZmljdWx0eUxldmVsLCAuLi5yZXN0IH0gPSByb3c7XG4gICAgICAgICAgY29uc3QgbmV3Um93OiBJU2NpZW5jZURhdGEgPSB7IC4uLnJlc3QsIERpZmZpY3VsdHlMZXZlbCB9O1xuXG4gICAgICAgICAgLy8gQnJlYWsgaW50byBjaHVua3NcbiAgICAgICAgICBsZXQgaW5kZXggPSAwO1xuICAgICAgICAgIHdoaWxlIChpbmRleCA8IG5ld1Jvdy5Ub3BpYy5sZW5ndGgpIHtcbiAgICAgICAgICAgIHNjaWVuY2VEYXRhQ2h1bmtzLnB1c2goeyAuLi5uZXdSb3csIElEOiB1dWlkKCksIFRvcGljOiBuZXdSb3cuVG9waWMuc2xpY2UoaW5kZXgsIGluZGV4ICsgY2h1bmtTaXplKSB9KTtcbiAgICAgICAgICAgIGluZGV4ICs9IGNodW5rU2l6ZTtcbiAgICAgICAgICB9XG4gICAgICAgIH0pO1xuXG4gICAgICAgIC8vIENyZWF0ZSBFbWJlZGRpbmdzXG4gICAgICAgIGNvbnN0IGVtYmVkZGluZ1Jlc3VsdHMgPSBhd2FpdCBvQWlTZGsuY3JlYXRlRW1iZWRkaW5ncyhzY2llbmNlRGF0YUNodW5rcywgZW1iZWRkaW5nQmF0Y2hTaXplKTtcblxuICAgICAgICAvLyBTdG9yZSBpbiBEQlxuICAgICAgICBjb25zdCByZXN1bHQgPSBhd2FpdCBJTlNFUlQuaW50byhTY2llbmNlRGF0YSkuZW50cmllcyhlbWJlZGRpbmdSZXN1bHRzKTtcblxuICAgICAgICByZXEucmVwbHkocmVzdWx0KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGxvZ2dlci5lcnJvcihcIkludmFsaWQgZmlsZSB0eXBlIG9yIG1pc3NpbmcgY29udGVudFwiKTtcbiAgICAgICAgcmVxLmVycm9yKHsgY29kZTogMSwgc3RhdHVzOiA0MDAsIG1lc3NhZ2U6IFwiSW52YWxpZCBmaWxlIHR5cGUgb3IgbWlzc2luZyBjb250ZW50XCIgfSk7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXgpIHtcbiAgICAgIGlmIChleCBpbnN0YW5jZW9mIEVycm9yKSB7XG4gICAgICAgIGxvZ2dlci5lcnJvcihgRXJyb3Igd2hpbGUgdXBsb2FkaW5nIGZpbGU6ICR7ZXgubWVzc2FnZX1gKTtcbiAgICAgICAgcmVxLmVycm9yKHsgY29kZTogMSwgc3RhdHVzOiA0MDAsIG1lc3NhZ2U6IGBFcnJvciB3aGlsZSB1cGxvYWRpbmcgZmlsZTogJHtleC5tZXNzYWdlfWAgfSk7XG4gICAgICB9XG4gICAgfVxuICB9O1xufVxuIl19