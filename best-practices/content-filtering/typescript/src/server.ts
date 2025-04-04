// server.tsx
import express, { Request, Response } from "express";
import { chatWithAgent, generateParaphrase } from "./services/aiOrchestration";
import { logger } from "./utils/logger";

const app = express();
app.use(express.json());
const port = process.env.PORT ?? 3000;

app.post(
  "/chatWithAgent",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { input, filterInput } = req.body;
      if (!input) {
        res.status(400).json({ error: "Input is required" });
        return;
      }

      try {
        const response = await chatWithAgent(input, filterInput);

        // Return response if no filtering was applied
        res.send(response.getContent());
      } catch (error: any) {
        logger.info(error);
        if (error.cause.status === 400) {
          logger.info("Input was filtered as expected.");
          res.status(400).json({ error: "Input was filtered" });
        } else {
          throw error;
        }
      }
    } catch (error) {
      logger.error(error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  }
);

app.post(
  "/generateParaphrase",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { input, filterOutput } = req.body;
      if (!input) {
        res.status(400).json({ error: "Input is required" });
        return;
      }

      const response = await generateParaphrase(input, filterOutput);

      // accessing the content should throw an error
      try {
        res.send(response.getContent());
      } catch (error: any) {
        logger.info(
          `Result from output content filter: ${
            response.data.module_results.output_filtering!.message
          }`
        );
        logger.info(
          "The original response from the LLM was as follows: " +
            response.data.module_results.llm?.choices[0].message.content
        );
        res.status(400).json({
          error: error.message,
          message: response.data.module_results.output_filtering!.message,
          llm_response:
            response.data.module_results.llm?.choices[0].message.content,
        });
      }
    } catch (error) {
      logger.error(error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  }
);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
