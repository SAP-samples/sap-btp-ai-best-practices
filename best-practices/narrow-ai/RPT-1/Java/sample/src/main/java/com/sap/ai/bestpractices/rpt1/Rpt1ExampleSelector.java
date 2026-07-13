package com.sap.ai.bestpractices.rpt1;

import java.util.List;

enum Rpt1ExampleSelector {
    WITHOUT_SCHEMA("without-schema"),
    WITH_SCHEMA("with-schema"),
    ALL("all");

    private final String cliValue;

    Rpt1ExampleSelector(String cliValue) {
        this.cliValue = cliValue;
    }

    static Rpt1ExampleSelector fromArgs(List<String> args) {
        if (args == null || args.isEmpty()) {
            return WITHOUT_SCHEMA;
        }

        String value = args.get(0).trim().toLowerCase();
        for (Rpt1ExampleSelector selector : values()) {
            if (selector.cliValue.equals(value)) {
                return selector;
            }
        }

        throw new IllegalArgumentException(
            "Unsupported example selector '%s'. Expected one of: without-schema, with-schema, all.".formatted(value)
        );
    }
}
