import { parse } from "csv-parse";
import { Readable } from "stream";
import { IncomingMessage } from "http";

export function parseFile(fileContent: IncomingMessage): Promise<any[]> {
  const results: any[] = [];

  return new Promise((resolve, reject) => {
    Readable.from(fileContent)
      .pipe(
        parse({
          delimiter: ",",
          columns: true, // If the CSV has a header row
          relax_quotes: true,
        })
      )
      .on("data", (data) => results.push(data))
      .on("end", () => {
        //fs.unlinkSync(filePath); // Delete the file after processing
        resolve(results);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}
