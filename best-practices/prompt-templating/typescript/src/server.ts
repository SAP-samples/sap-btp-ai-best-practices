// server.tsx
import express, { Request, Response } from "express";
import { askCapitalOfCountry } from "./services/aiOrchestration";
import { logger } from "./utils/logger";

const app = express();
app.use(express.json());
const port = process.env.PORT ?? 3000;

app.post(
  "/askCapitalOfCountry",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { country } = req.body;
      if (!country) {
        res.status(400).json({ error: "Country is required" });
        return;
      }

      const response = await askCapitalOfCountry(country);

      logger.info(response.getContent());
      res.send(response.getContent());
    } catch (error) {
      logger.error(error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  }
);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
