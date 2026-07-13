package com.sap.ai.bestpractices.rpt1;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;

class Rpt1ConfigTest {

    @Test
    void fromEnvironmentUsesDefaults() {
        MockEnvironment environment = new MockEnvironment()
            .withProperty("AICORE_SERVICE_KEY", "{}");

        Rpt1Config config = Rpt1Config.fromEnvironment(environment);

        assertEquals("sap-rpt-1-small", config.model().name());
    }

    @Test
    void fromEnvironmentRequiresServiceKey() {
        MockEnvironment environment = new MockEnvironment();

        assertThrows(IllegalArgumentException.class, () -> Rpt1Config.fromEnvironment(environment));
    }
}
