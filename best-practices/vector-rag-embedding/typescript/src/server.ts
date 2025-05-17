// server.tsx
import express, { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";
dotenv.config();

// Extend the Request interface to include the file property added by multer
declare global {
  namespace Express {
    interface Request {
      file?: Express.Multer.File;
    }
  }
}
import { logger } from "./utils/logger";
import { parseFile } from "./utils/csv-file";
import AiSdk from "./services/ai-sdk";
import HANADb from "./services/hana-db";
import { IScienceData } from "./types";

const oAiSdk = new AiSdk();
const oHanaDb = new HANADb();
const multer = require("multer");
const upload = multer({ dest: "uploads/" });
const app = express();
const port = process.env.PORT ?? 3000;
const chunkSize = 500;
const embeddingBatchSize = 100;

app.use(express.json());

// Handle Requests
app.get("/scienceData", async (req: Request, res: Response): Promise<void> => {
  // Query the database
  const result = await oHanaDb.executeQuery(`SELECT ID,TOPIC,DIFFICULTY_LEVEL,CATEGORY FROM AI_DB_SCIENCE_DATA;`);
  // Return the results
  res.status(200).json({ count: result.length, result });
});

app.post("/uploadScienceData", upload.single("csvFile"), async (req: Request, res: Response): Promise<void> => {
  try {
    const filePath = req.file?.path;

    if (!filePath) {
      res.status(400).send("No file uploaded.");
    } else {
      // Parse the CSV file
      const scienceDataRows: any[] = await parseFile(filePath);

      // Break into chunks
      let scienceDataChunks: any[] = [];
      scienceDataRows.forEach((row: IScienceData) => {
        let index = 0;
        while (index < row.Topic.length) {
          scienceDataChunks.push({ ...row, ID: uuidv4(), Topic: row.Topic.slice(index, index + chunkSize) });
          index += chunkSize;
        }
      });

      // Create Embeddings
      const embeddingResults = await oAiSdk.createEmbeddings(scienceDataChunks, embeddingBatchSize);

      // Store in DB
      const columns = "TOPIC, DIFFICULTY_LEVEL, CATEGORY, ID, EMBEDDING";
      const values = embeddingResults.map((row: any) => Object.values(row));
      const result = await oHanaDb.executeBatchQuery(
        `INSERT INTO AI_DB_SCIENCE_DATA (${columns}) VALUES (?,?,?,?,TO_REAL_VECTOR(?))`,
        values
      );

      // Return the results
      res.status(200).json(result);
    }
  } catch (error) {
    logger.error(error);
    res.status(500).json({
      error: "Internal Server Error",
      message: error instanceof Error ? error.message : "An unknown error occurred",
    });
  }
});

// Start Server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
