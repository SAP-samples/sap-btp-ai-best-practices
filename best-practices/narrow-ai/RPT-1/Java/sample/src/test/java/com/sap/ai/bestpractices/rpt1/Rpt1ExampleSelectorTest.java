package com.sap.ai.bestpractices.rpt1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class Rpt1ExampleSelectorTest {

    @Test
    void defaultsToWithoutSchema() {
        assertEquals(Rpt1ExampleSelector.WITHOUT_SCHEMA, Rpt1ExampleSelector.fromArgs(List.of()));
    }

    @Test
    void parsesSupportedValues() {
        assertEquals(Rpt1ExampleSelector.WITH_SCHEMA, Rpt1ExampleSelector.fromArgs(List.of("with-schema")));
        assertEquals(Rpt1ExampleSelector.ALL, Rpt1ExampleSelector.fromArgs(List.of("all")));
    }

    @Test
    void rejectsUnsupportedValues() {
        assertThrows(IllegalArgumentException.class, () -> Rpt1ExampleSelector.fromArgs(List.of("nope")));
    }
}
