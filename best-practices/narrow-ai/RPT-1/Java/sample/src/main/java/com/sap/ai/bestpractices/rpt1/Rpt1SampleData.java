package com.sap.ai.bestpractices.rpt1;

import com.sap.ai.sdk.foundationmodels.rpt.generated.model.ColumnType;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.PredictionPlaceholder;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.RowsInnerValue;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.SchemaFieldConfig;
import com.sap.ai.sdk.foundationmodels.rpt.generated.model.TargetColumnConfig;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

final class Rpt1SampleData {

    private Rpt1SampleData() {
    }

    static List<Map<String, RowsInnerValue>> classificationRows() {
        return List.of(
            Map.of(
                "PRODUCT", RowsInnerValue.create("Couch"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(999.99)),
                "ORDERDATE", RowsInnerValue.create("2025-11-28"),
                "ID", RowsInnerValue.create("35"),
                "COSTCENTER", RowsInnerValue.create("[PREDICT]")
            ),
            Map.of(
                "PRODUCT", RowsInnerValue.create("Office Chair"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(150.80)),
                "ORDERDATE", RowsInnerValue.create("2025-11-02"),
                "ID", RowsInnerValue.create("44"),
                "COSTCENTER", RowsInnerValue.create("Office Furniture")
            ),
            Map.of(
                "PRODUCT", RowsInnerValue.create("Server Rack"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(2200.00)),
                "ORDERDATE", RowsInnerValue.create("2025-11-01"),
                "ID", RowsInnerValue.create("104"),
                "COSTCENTER", RowsInnerValue.create("Data Infrastructure")
            )
        );
    }

    static List<Map<String, RowsInnerValue>> discountRows() {
        return List.of(
            Map.of(
                "PRODUCT", RowsInnerValue.create("Standing Desk"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(780.00)),
                "ORDERDATE", RowsInnerValue.create("2025-11-30"),
                "ID", RowsInnerValue.create("88"),
                "DISCOUNT_RATE", RowsInnerValue.create(BigDecimal.valueOf(-1))
            ),
            Map.of(
                "PRODUCT", RowsInnerValue.create("Office Chair"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(150.80)),
                "ORDERDATE", RowsInnerValue.create("2025-11-02"),
                "ID", RowsInnerValue.create("44"),
                "DISCOUNT_RATE", RowsInnerValue.create(BigDecimal.valueOf(0.05))
            ),
            Map.of(
                "PRODUCT", RowsInnerValue.create("Server Rack"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(2200.00)),
                "ORDERDATE", RowsInnerValue.create("2025-11-01"),
                "ID", RowsInnerValue.create("104"),
                "DISCOUNT_RATE", RowsInnerValue.create(BigDecimal.valueOf(0.12))
            ),
            Map.of(
                "PRODUCT", RowsInnerValue.create("Monitor Arm"),
                "PRICE", RowsInnerValue.create(BigDecimal.valueOf(115.20)),
                "ORDERDATE", RowsInnerValue.create("2025-11-08"),
                "ID", RowsInnerValue.create("52"),
                "DISCOUNT_RATE", RowsInnerValue.create(BigDecimal.valueOf(0.04))
            )
        );
    }

    static Map<String, SchemaFieldConfig> discountSchema() {
        return Map.of(
            "PRODUCT", SchemaFieldConfig.create().dtype(ColumnType.STRING),
            "PRICE", SchemaFieldConfig.create().dtype(ColumnType.NUMERIC),
            "ORDERDATE", SchemaFieldConfig.create().dtype(ColumnType.DATE),
            "ID", SchemaFieldConfig.create().dtype(ColumnType.STRING),
            "DISCOUNT_RATE", SchemaFieldConfig.create().dtype(ColumnType.NUMERIC)
        );
    }

    static List<TargetColumnConfig> classificationTargetColumns() {
        return List.of(
            TargetColumnConfig.create()
                .name("COSTCENTER")
                .predictionPlaceholder(PredictionPlaceholder.create("[PREDICT]"))
                .taskType(TargetColumnConfig.TaskTypeEnum.CLASSIFICATION)
        );
    }

    static List<TargetColumnConfig> discountTargetColumns() {
        return List.of(
            TargetColumnConfig.create()
                .name("DISCOUNT_RATE")
                .predictionPlaceholder(PredictionPlaceholder.create(BigDecimal.valueOf(-1)))
                .taskType(TargetColumnConfig.TaskTypeEnum.REGRESSION)
        );
    }
}
