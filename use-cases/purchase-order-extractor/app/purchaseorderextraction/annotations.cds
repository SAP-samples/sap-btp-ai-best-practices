using DocumentService as service from '../../srv/service';
using from '../../db/schema';

annotate service.PurchaseOrders with @(
    UI.FieldGroup #GeneratedGroup : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Label : 'senderPostalCode',
                Value : senderPostalCode,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderState',
                Value : senderState,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderStreet',
                Value : senderStreet,
            },
            {
                $Type : 'UI.DataField',
                Label : 'documentDate',
                Value : documentDate,
            },
            {
                $Type : 'UI.DataField',
                Label : 'documentNumber',
                Value : documentNumber,
            },
            {
                $Type : 'UI.DataField',
                Label : 'grossAmount',
                Value : grossAmount,
            },
            {
                $Type : 'UI.DataField',
                Label : 'netAmount',
                Value : netAmount,
            },
            {
                $Type : 'UI.DataField',
                Label : 'paymentTerms',
                Value : paymentTerms,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderAddress',
                Value : senderAddress,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderCity',
                Value : senderCity,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderCountryCode',
                Value : senderCountryCode,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderFax',
                Value : senderFax,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderId',
                Value : senderId,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderName',
                Value : senderName,
            },
            {
                $Type : 'UI.DataField',
                Label : 'senderPhone',
                Value : senderPhone,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToAddress',
                Value : shipToAddress,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToCity',
                Value : shipToCity,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToCountryCode',
                Value : shipToCountryCode,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToFax',
                Value : shipToFax,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToName',
                Value : shipToName,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToPhone',
                Value : shipToPhone,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToPostalCode',
                Value : shipToPostalCode,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToState',
                Value : shipToState,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shipToStreet',
                Value : shipToStreet,
            },
            {
                $Type : 'UI.DataField',
                Label : 'shippingTerms',
                Value : shippingTerms,
            },
            {
                $Type : 'UI.DataField',
                Label : 'extractionReviewStatus',
                Value : extractionReviewStatus,
            },
            {
                $Type : 'UI.DataField',
                Label : 'paymentStatus',
                Value : paymentStatus,
            },
            {
                $Type : 'UI.DataField',
                Label : 'filename',
                Value : filename,
            },
            {
                $Type : 'UI.DataField',
                Label : 'aiExtractionReview',
                Value : aiExtractionReview,
            },
        ],
    },
    UI.Facets : [
        {
            $Type : 'UI.CollectionFacet',
            Label : 'Customer',
            ID : 'Customer1',
            Facets : [
                {
                    $Type : 'UI.ReferenceFacet',
                    Label : 'Customer Matching',
                    ID : 'CustomerName',
                    Target : '@UI.FieldGroup#CustomerName',
                },
                {
                    $Type : 'UI.ReferenceFacet',
                    Label : 'Ship To',
                    ID : 'ShipTo',
                    Target : '@UI.FieldGroup#ShipTo',
                },
            ],
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'GeneratedFacet1',
            Label : 'General Information',
            Target : '@UI.FieldGroup#GeneratedGroup',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Extracted Line Items',
            ID : 'LineItems',
            Target : 'lineItems/@UI.LineItem#LineItems',
        },
    ],
    UI.LineItem : [
        {
            $Type : 'UI.DataField',
            Label : 'Document Date',
            Value : documentDate,
        },
        {
            $Type : 'UI.DataField',
            Label : 'Document Number',
            Value : documentNumber,
        },
        {
            $Type : 'UI.DataField',
            Value : filename,
            Label : 'File Name',
        },
        {
            $Type : 'UI.DataField',
            Value : customer.name,
        },
        {
            $Type : 'UI.DataFieldForAnnotation',
            Target : '@UI.DataPoint#lineItemsWithSapMaterialCount',
            Label : 'Material Matching Progress',
        },
    ],
    UI.FieldGroup #Customer : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : customer_ID,
                Label : 'Customer Number',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.name,
            },
            {
                $Type : 'UI.DataField',
                Value : customerReason,
                Label : 'Customer Reason',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.street,
                Label : 'street',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.postalCode,
                Label : 'postalCode',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.city,
                Label : 'city',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.region,
                Label : 'region',
            },
        ],
    },
    UI.DataPoint #lineItemsWithSapMaterialCount : {
        Value : lineItemsWithSapMaterialCount,
        Visualization : #Progress,
        TargetValue : totalLineItemsCount,
    },
    UI.SelectionPresentationVariant #table : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.LineItem',
            ],
            SortOrder : [
                {
                    $Type : 'Common.SortOrderType',
                    Property : createdAt,
                    Descending : true,
                },
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
            ],
        },
    },
    UI.HeaderInfo : {
        Title : {
            $Type : 'UI.DataField',
            Value : filename,
        },
        TypeName : 'Purchase Order',
        TypeNamePlural : 'Purchase Orders',
    },
    UI.Identification : [
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'DocumentService.file',
            Label : 'file',
        },
    ],
    UI.FieldGroup #CustomerName : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : customer.ID,
            },
            {
                $Type : 'UI.DataField',
                Value : customerReason,
            },
        ],
    },
    UI.FieldGroup #ShipTo : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : customer.street,
                Label : 'street',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.city,
                Label : 'city',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.postalCode,
                Label : 'postalCode',
            },
            {
                $Type : 'UI.DataField',
                Value : customer.region,
                Label : 'region',
            },
        ],
    },
);

annotate service.LineItems with @(
    UI.LineItem #LineItems : [
        {
            $Type : 'UI.DataField',
            Value : description,
            Label : 'Description',
            Criticality : matchingStatus_criticality,
        },
        {
            $Type : 'UI.DataField',
            Value : customerMaterialNumber,
            Label : 'Customer Material Number',
        },
        {
            $Type : 'UI.DataField',
            Value : supplierMaterialNumber,
            Label : 'Supplier Material Number',
        },
        {
            $Type : 'UI.DataField',
            Value : netAmount,
            Label : 'Net Amount',
        },
    ],
    UI.Facets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Matched Material',
            ID : 'MatchedMaterial',
            Target : '@UI.FieldGroup#MatchedMaterial',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Material Matching Candidates',
            ID : 'MaterialMatchingCandidates',
            Target : 'mappingCandidates/@UI.LineItem#MaterialMatchingCandidates',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Material Search',
            ID : 'MaterialSearch',
            Target : 'allMaterials/@UI.LineItem#MaterialSearch',
        },
    ],
    UI.FieldGroup #MatchedMaterial : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : sapMaterialNumber,
                Label : 'sapMaterialNumber',
            },
            {
                $Type : 'UI.DataField',
                Value : materialSelectionReason,
                Label : 'materialSelectionReason',
            },
        ],
    },
    UI.HeaderInfo : {
        TypeName : 'Line Item',
        TypeNamePlural : 'Line Item',
        Title : {
            $Type : 'UI.DataField',
            Value : description,
        },
    },
    UI.DataPoint #lineNumber : {
        $Type : 'UI.DataPointType',
        Value : lineNumber,
        Title : 'Extracted Line Number',
    },
    UI.DataPoint #description1 : {
        $Type : 'UI.DataPointType',
        Value : description,
        Title : 'Extracted Description',
    },
    UI.DataPoint #customerMaterialNumber1 : {
        $Type : 'UI.DataPointType',
        Value : customerMaterialNumber,
        Title : 'Extracted Customer Material #',
    },
    UI.DataPoint #supplierMaterialNumber1 : {
        $Type : 'UI.DataPointType',
        Value : supplierMaterialNumber,
        Title : 'Extracted Supplier Material #',
    },
    UI.DataPoint #unitPrice1 : {
        $Type : 'UI.DataPointType',
        Value : unitPrice,
        Title : 'Extracted Unit Price',
    },
    UI.DataPoint #quantity1 : {
        $Type : 'UI.DataPointType',
        Value : quantity,
        Title : 'Extracted Quantity',
    },
    UI.DataPoint #netAmount : {
        $Type : 'UI.DataPointType',
        Value : netAmount,
        Title : 'Extracted Net Amount',
    },
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'lineNumber',
            Target : '@UI.DataPoint#lineNumber',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'description',
            Target : '@UI.DataPoint#description1',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'customerMaterialNumber',
            Target : '@UI.DataPoint#customerMaterialNumber1',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'supplierMaterialNumber',
            Target : '@UI.DataPoint#supplierMaterialNumber1',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'unitPrice',
            Target : '@UI.DataPoint#unitPrice1',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'quantity',
            Target : '@UI.DataPoint#quantity1',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'netAmount',
            Target : '@UI.DataPoint#netAmount',
        },
    ],
);

annotate service.PurchaseOrders with {
    customer @(
        Common.ValueList : {
            $Type : 'Common.ValueListType',
            CollectionPath : 'Customers',
            Parameters : [
                {
                    $Type : 'Common.ValueListParameterInOut',
                    LocalDataProperty : customer_ID,
                    ValueListProperty : 'ID',
                },
            ],
        },
        Common.ValueListWithFixedValues : true,
)};

annotate service.Customers with {
    ID @(
        Common.Text : name,
        Common.Text.@UI.TextArrangement : #TextLast,
    )
};

annotate service.Customers with {
    name @Common.FieldControl : #ReadOnly
};
annotate service.LineItemCMIRCandidates with @(
    UI.LineItem #MaterialMatchingCandidates : [
        {
            $Type : 'UI.DataField',
            Value : lineitem.customerMaterialNumber,
            Label : 'Extracted Customer Mat. #',
        },
        {
            $Type : 'UI.DataField',
            Value : mapping.customerMaterialNumber,
            Label : 'CMIR Customer Mat. #',
        },
        {
            $Type : 'UI.DataField',
            Value : lineitem.supplierMaterialNumber,
            Label : 'Extracted SAP Mat. #',
        },
        {
            $Type : 'UI.DataField',
            Value : mapping.supplierMaterialNumber,
            Label : 'CMIR SAP Mat. #',
        },
        {
            $Type : 'UI.DataField',
            Value : lineitem.description,
            Label : 'Extracted Description',
        },
        {
            $Type : 'UI.DataField',
            Value : mapping.materialDescription,
            Label : 'SAP Description',
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'DocumentService.selectMaterial',
            Label : 'Select Material',
            Inline : true,
        },
    ]
);

annotate service.V_Materials with @(
    UI.LineItem #MaterialSearch : [
        {
            $Type : 'UI.DataField',
            Value : supplierMaterialNumber,
        },
        {
            $Type : 'UI.DataField',
            Value : materialDescription,
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'DocumentService.matchSearchMaterial',
            Label : 'Match',
            Inline : true,
        },
    ]
);

