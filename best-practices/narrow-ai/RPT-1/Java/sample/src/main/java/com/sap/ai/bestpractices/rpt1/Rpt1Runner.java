package com.sap.ai.bestpractices.rpt1;

import com.sap.ai.sdk.foundationmodels.rpt.RptClient;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.PredictRequestPayload;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.PredictResponsePayload;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.PredictionConfig;
import java.util.List;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
class Rpt1Runner {

    private final Environment environment;

    Rpt1Runner(Environment environment) {
        this.environment = environment;
    }

    void run(List<String> args) {
        Rpt1Config config = Rpt1Config.fromEnvironment(environment);
        Rpt1ExampleSelector selector = Rpt1ExampleSelector.fromArgs(args);

        if (selector == Rpt1ExampleSelector.WITHOUT_SCHEMA || selector == Rpt1ExampleSelector.ALL) {
            runWithoutSchema(config);
        }

        if (selector == Rpt1ExampleSelector.WITH_SCHEMA || selector == Rpt1ExampleSelector.ALL) {
            runWithSchema(config);
        }
    }

    private void runWithoutSchema(Rpt1Config config) {
        var client = RptClient.forModel(config.model());
        var input = PredictRequestPayload.create()
            .predictionConfig(PredictionConfig.create().targetColumns(Rpt1SampleData.classificationTargetColumns()))
            .indexColumn("ID")
            .rows(Rpt1SampleData.classificationRows());

        var response = client.tableCompletion(input);
        printResult("without-schema", response);
    }

    private void runWithSchema(Rpt1Config config) {
        var client = RptClient.forModel(config.model());
        var input = PredictRequestPayload.create()
            .predictionConfig(PredictionConfig.create().targetColumns(Rpt1SampleData.discountTargetColumns()))
            .indexColumn("ID")
            .dataSchema(Rpt1SampleData.discountSchema())
            .parseDataTypes(true)
            .rows(Rpt1SampleData.discountRows());

        var response = client.tableCompletion(input);
        printResult("with-schema", response);
    }

    private void printResult(String exampleName, PredictResponsePayload response) {
        System.out.println("=== " + exampleName + " ===");
        System.out.println("requestId=" + response.getId());
        System.out.println("status=" + response.getStatus());
        if (!response.getPredictions().isEmpty()) {
            System.out.println("firstPrediction=" + response.getPredictions().get(0));
        }
        System.out.println("metadata=" + response.getMetadata());
    }
}
