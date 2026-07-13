package com.sap.ai.bestpractices.rpt1;

import com.sap.ai.sdk.foundationmodels.rpt.RptModel;
import org.springframework.core.env.Environment;

record Rpt1Config(RptModel model) {

    private static final String DEFAULT_MODEL_NAME = "sap-rpt-1-small";

    static Rpt1Config fromEnvironment(Environment environment) {
        required(environment, "AICORE_SERVICE_KEY");
        String modelName = environment.getProperty("RPT1_MODEL_NAME", DEFAULT_MODEL_NAME);
        String modelVersion = environment.getProperty("RPT1_MODEL_VERSION");
        return new Rpt1Config(resolveModel(modelName, modelVersion));
    }

    private static String required(Environment environment, String key) {
        String value = environment.getProperty(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing required environment variable: " + key);
        }
        return value;
    }

    private static RptModel resolveModel(String modelName, String modelVersion) {
        RptModel model = switch (modelName) {
            case "sap-rpt-1-small" -> RptModel.SAP_RPT_1_SMALL;
            case "sap-rpt-1-large" -> RptModel.SAP_RPT_1_LARGE;
            default -> new RptModel(modelName, null);
        };

        if (modelVersion == null || modelVersion.isBlank()) {
            return model;
        }

        return model.withVersion(modelVersion);
    }
}
