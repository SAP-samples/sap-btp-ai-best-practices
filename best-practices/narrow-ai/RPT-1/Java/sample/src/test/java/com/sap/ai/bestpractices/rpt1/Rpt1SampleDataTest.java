package com.sap.ai.bestpractices.rpt1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class Rpt1SampleDataTest {

    @Test
    void classificationRowsUsePredictPlaceholder() {
        var row = Rpt1SampleData.classificationRows().get(0);
        assertTrue(row.get("COSTCENTER").toString().contains("[PREDICT]"));
    }

    @Test
    void structuredTargetUsesDiscountRate() {
        assertEquals("DISCOUNT_RATE", Rpt1SampleData.discountTargetColumns().get(0).getName());
    }
}
