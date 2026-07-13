sap.ui.define([
    "sap/fe/test/JourneyRunner",
	"purchaseorderextraction/test/integration/pages/PurchaseOrdersList",
	"purchaseorderextraction/test/integration/pages/PurchaseOrdersObjectPage",
	"purchaseorderextraction/test/integration/pages/LineItemsObjectPage"
], function (JourneyRunner, PurchaseOrdersList, PurchaseOrdersObjectPage, LineItemsObjectPage) {
    'use strict';

    var runner = new JourneyRunner({
        launchUrl: sap.ui.require.toUrl('purchaseorderextraction') + '/test/flp.html#app-preview',
        pages: {
			onThePurchaseOrdersList: PurchaseOrdersList,
			onThePurchaseOrdersObjectPage: PurchaseOrdersObjectPage,
			onTheLineItemsObjectPage: LineItemsObjectPage
        },
        async: true
    });

    return runner;
});

