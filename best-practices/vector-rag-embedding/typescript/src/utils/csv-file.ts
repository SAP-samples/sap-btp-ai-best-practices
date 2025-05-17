import fs from "fs";
import { parse } from "csv-parse";

export function parseFile(filePath: string): Promise<string[]> {
  const results: any[] = [];

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(
        parse({
          delimiter: ",",
          columns: true, // If the CSV has a header row
        })
      )
      .on("data", (data) => results.push(data))
      .on("end", () => {
        fs.unlinkSync(filePath); // Delete the file after processing
        resolve(results);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}
