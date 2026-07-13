sap.ui.define([
    "sap/fe/test/JourneyRunner",
	"posomaterialmapping/test/integration/pages/LineItemSalesOrderItemCandidatesList",
	"posomaterialmapping/test/integration/pages/LineItemSalesOrderItemCandidatesObjectPage"
], function (JourneyRunner, LineItemSalesOrderItemCandidatesList, LineItemSalesOrderItemCandidatesObjectPage) {
    'use strict';

    var runner = new JourneyRunner({
        launchUrl: sap.ui.require.toUrl('posomaterialmapping') + '/test/flp.html#app-preview',
        pages: {
			onTheLineItemSalesOrderItemCandidatesList: LineItemSalesOrderItemCandidatesList,
			onTheLineItemSalesOrderItemCandidatesObjectPage: LineItemSalesOrderItemCandidatesObjectPage
        },
        async: true
    });

    return runner;
});

