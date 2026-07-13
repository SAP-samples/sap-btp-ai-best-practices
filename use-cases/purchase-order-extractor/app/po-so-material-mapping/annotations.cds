using DocumentService as service from '../../srv/service';
annotate service.LineItemSalesOrderItemCandidates with @(
    UI.FieldGroup #GeneratedGroup : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : reason,
            },
        ],
    },
    UI.Facets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Matched Sales Order Item',
            ID : 'SalesOrder',
            Target : '@UI.FieldGroup#SalesOrder',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Sales Order Items Associated with PO',
            ID : 'SalesOrderItemsAssociatedwithPO',
            Target : 'toSoItemsByPo/@UI.LineItem#SalesOrderItemsAssociatedwithPO',
        },
    ],
    UI.LineItem : [
        {
            $Type : 'UI.DataField',
            Value : customer,
        },
        {
            $Type : 'UI.DataField',
            Value : poNumber,
            Label : 'Purchase Order #',
        },
        {
            $Type : 'UI.DataField',
            Value : purchaseOrderLineItem.customerMaterialNumber,
            Label : 'Customer Material #',
        },
        {
            $Type : 'UI.DataField',
            Value : purchaseOrderLineItem.description,
            Label : 'Description',
        },
        {
            $Type : 'UI.DataField',
            Value : salesOrderLineItem.salesOrderNumber,
            Label : 'Sales Order #',
        },
        {
            $Type : 'UI.DataField',
            Value : salesOrderLineItem.salesOrderItemText,
            Label : 'Sales Order Item Text',
        },
        {
            $Type : 'UI.DataField',
            Value : reason,
            Criticality : matchingStatus_criticality,
        },
    ],
    UI.SelectionPresentationVariant #table : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.LineItem',
            ],
            GroupBy : [
                poNumber,
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
            ],
        },
    },
    UI.SelectionFields : [
        hasMatch,
        customer,
        poNumber,
        soNumber,
    ],
    UI.DataPoint #reason : {
        $Type : 'UI.DataPointType',
        Value : reason,
        Title : 'Reason',
    },
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Extracted Line Item',
            ID : 'ExtractedLineItem',
            Target : '@UI.FieldGroup#ExtractedLineItem',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'reason',
            Target : '@UI.DataPoint#reason',
        },
    ],
    UI.FieldGroup #SalesOrder : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.lineNumber,
                Label : 'lineNumber',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.dateCreated,
                Label : 'dateCreated',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.customerMaterialNumber,
                Label : 'customerMaterialNumber',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.salesOrderNumber,
                Label : 'salesOrderNumber',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.salesOrderItemText,
                Label : 'salesOrderItemText',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.sapMaterialNumber,
                Label : 'sapMaterialNumber',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.shipTo,
                Label : 'shipTo',
            },
            {
                $Type : 'UI.DataField',
                Value : salesOrderLineItem.soldTo,
                Label : 'soldTo',
            },
        ],
    },
    UI.FieldGroup #ExtractedLineItem : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.lineNumber,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.customerMaterialNumber,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.supplierMaterialNumber,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.unitPrice,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.quantity,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.description,
            },
            {
                $Type : 'UI.DataField',
                Value : purchaseOrderLineItem.netAmount,
            },
        ],
    },
);

annotate service.LineItemSalesOrderItemCandidates with {
    purchaseOrderLineItem @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'LineItems',
        Parameters : [
            {
                $Type : 'Common.ValueListParameterInOut',
                LocalDataProperty : purchaseOrderLineItem_ID,
                ValueListProperty : 'ID',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'description',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'netAmount',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'quantity',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'unitPrice',
            },
        ],
    }
};

annotate service.LineItemSalesOrderItemCandidates with {
    salesOrderLineItem @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'SalesOrderItems',
        Parameters : [
            {
                $Type : 'Common.ValueListParameterInOut',
                LocalDataProperty : salesOrderLineItem_ID,
                ValueListProperty : 'ID',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'salesOrderNumber',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'lineNumber',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'dateCreated',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'sapMaterialNumber',
            },
        ],
    }
};

annotate service.LineItemSalesOrderItemCandidates with {
    customer @Common.Label : 'Customer'
};

annotate service.LineItemSalesOrderItemCandidates with {
    hasMatch @(
        Common.Label : 'hasMatch',
        )
};

annotate service.LineItems with @(
    UI.DataPoint #customerMaterialNumber : {
        $Type : 'UI.DataPointType',
        Value : customerMaterialNumber,
        Title : 'customerMaterialNumber',
    },
    UI.DataPoint #supplierMaterialNumber : {
        $Type : 'UI.DataPointType',
        Value : supplierMaterialNumber,
        Title : 'supplierMaterialNumber',
    },
    UI.DataPoint #description : {
        $Type : 'UI.DataPointType',
        Value : description,
        Title : 'description',
    },
    UI.DataPoint #unitPrice : {
        $Type : 'UI.DataPointType',
        Value : unitPrice,
        Title : 'unitPrice',
    },
    UI.DataPoint #quantity : {
        $Type : 'UI.DataPointType',
        Value : quantity,
        Title : 'quantity',
    },
);

annotate service.V_SalesOrderItemsByPoId with @(
    UI.LineItem #SalesOrderItemsAssociatedwithPO : [
        {
            $Type : 'UI.DataField',
            Value : salesOrderNumber,
        },
        {
            $Type : 'UI.DataField',
            Value : lineNumber,
        },
        {
            $Type : 'UI.DataField',
            Value : sapMaterialNumber,
        },
        {
            $Type : 'UI.DataField',
            Value : salesOrderItemText,
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'DocumentService.match',
            Label : 'Match',
            Inline : true,
        },
    ]
);

annotate service.LineItemSalesOrderItemCandidates with {
    poNumber @(
        Common.ValueList : {
            $Type : 'Common.ValueListType',
            CollectionPath : 'PurchaseOrders',
            Parameters : [
                {
                    $Type : 'Common.ValueListParameterInOut',
                    LocalDataProperty : poNumber,
                    ValueListProperty : 'documentNumber',
                },
                {
                    $Type : 'Common.ValueListParameterDisplayOnly',
                    ValueListProperty : 'filename',
                },
            ],
        },
        Common.ValueListWithFixedValues : true,
)};

