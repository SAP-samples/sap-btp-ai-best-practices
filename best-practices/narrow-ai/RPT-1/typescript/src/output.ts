type PrintablePredictionResponse = {
  id: string;
  status: {
    code: number;
    message: string;
  };
  predictions: Array<Record<string, unknown>>;
  metadata: {
    num_predictions: number;
  };
};

export function printResult(exampleName: string, response: PrintablePredictionResponse): void {
  console.log(`=== ${exampleName} ===`);
  console.log(`requestId: ${response.id}`);
  console.log(`status: ${response.status.code} ${response.status.message}`);
  if (response.predictions.length > 0) {
    console.log("firstPrediction:");
    console.log(JSON.stringify(response.predictions[0], null, 2));
  }
  console.log(`numPredictions: ${response.metadata.num_predictions}`);
}
