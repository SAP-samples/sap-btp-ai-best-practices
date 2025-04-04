// server.tsx
import express, { Request, Response } from "express";
import { orchestrationChatCompletionMasking } from "./services/aiOrchestration";
import { logger } from "./utils/logger";

const app = express();
app.use(express.json());
const port = process.env.PORT ?? 3000;

app.post("/generateEmail", async (req: Request, res: Response): Promise<void> => {
  try {
    const { prompt, anonymize = false } = req.body;
    if (!prompt) {
      res.status(400).json({ error: "Prompt is required" });
      return;
    }

    const response1 = await orchestrationChatCompletionMasking(prompt, anonymize);

    res.json({
      prompt: prompt,
      anonymize: anonymize,
      response: response1.getContent(),
    });
  } catch (error) {
    logger.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
