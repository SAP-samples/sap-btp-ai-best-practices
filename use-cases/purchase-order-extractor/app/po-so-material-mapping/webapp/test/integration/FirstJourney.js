sap.ui.define([
    "sap/ui/test/opaQunit",
    "./pages/JourneyRunner"
], function (opaTest, runner) {
    "use strict";

    function journey() {
        QUnit.module("First journey");

        opaTest("Start application", function (Given, When, Then) {
            Given.iStartMyApp();

            Then.onTheLineItemSalesOrderItemCandidatesList.iSeeThisPage();

        });


        opaTest("Navigate to ObjectPage", function (Given, When, Then) {
            // Note: this test will fail if the ListReport page doesn't show any data
            
            When.onTheLineItemSalesOrderItemCandidatesList.onFilterBar().iExecuteSearch();
            
            Then.onTheLineItemSalesOrderItemCandidatesList.onTable().iCheckRows();

            When.onTheLineItemSalesOrderItemCandidatesList.onTable().iPressRow(0);
            Then.onTheLineItemSalesOrderItemCandidatesObjectPage.iSeeThisPage();

        });

        opaTest("Teardown", function (Given, When, Then) { 
            // Cleanup
            Given.iTearDownMyApp();
        });
    }

    runner.run([journey]);
});