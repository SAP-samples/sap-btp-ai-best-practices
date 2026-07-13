using { com.sap.btp.ai as my } from '../db/schema.cds';

@path : '/service/DocumentService'
service DocumentService
{
    @odata.draft.enabled
    @readonly
    entity PurchaseOrders as projection on my.PurchaseOrders {
        *,
        // Calculated count properties for line items
        virtual totalLineItemsCount : Integer,
        virtual lineItemsWithSapMaterialCount : Integer,
        virtual lineItemsWithoutSapMaterialCount : Integer,
        @Core.MediaType: 'application/pdf'
        @Core.IsMediaType: true
        virtual pdfContent : LargeBinary
    };

    entity LineItems as projection on my.LineItems {
        *,
        case
      when sapMaterialNumber is not null then 3
      when sapMaterialNumber is null then 2
      else 1
    end as matchingStatus_criticality : Integer,
    allMaterials : Association to many V_Materials on true // all
    };

    entity LineItemCMIRCandidates as projection on my.LineItemCMIRCandidates {
        *
    } actions {
        action selectMaterial();
    };

    @cds.redirection.target
    entity CMIRMappings as projection on my.CMIRMappings;

    entity Customers as projection on my.Customers;

    entity SalesOrders as projection on my.SalesOrders;

    entity SalesOrderItems as projection on my.SalesOrderItems;

    entity V_SalesOrderItemsByPoId as select from my.SalesOrders as so
        inner join my.PurchaseOrders  as po  on po.documentNumber   = so.purchaseOrderNumber
        inner join my.SalesOrderItems as soi on soi.salesOrderNumber = so.salesOrderNumber
        {
        key po.ID   as po_id,       
        key soi.ID  as so_item_id,

        soi.salesOrderNumber,
        soi.lineNumber,
        soi.sapMaterialNumber,
        soi.salesOrderItemText
        } actions {
            action match();
        };

    entity V_Materials as projection on my.CMIRMappings {
        key ID,                                
        supplierMaterialNumber,
        materialDescription,
       
        toParent : Association to LineItems
            on toParent.ID = $self.ID

    } actions {
        @( cds.odata.bindingparameter.name : '_it', Common.SideEffects : {TargetProperties : ['_it/toParent']} )
        action matchSearchMaterial(
            @mandatory
            @UI.ParameterDefaultValue : true
            @Common.Label: 'Create CMIR record?'
            createCMIR: Boolean
        ) returns String; 
    }

    @readonly
    entity LineItemSalesOrderItemCandidates as projection on my.LineItemSalesOrderItemCandidates {
        *,
        purchaseOrderLineItem.purchaseOrder.customer.name as customer,
        purchaseOrderLineItem.purchaseOrder.documentNumber as poNumber,
        purchaseOrderLineItem.purchaseOrder.ID as poID,
        salesOrderLineItem.salesOrder.salesOrderNumber as soNumber,
        salesOrderLineItem.sapMaterialNumber as sapMaterialNumber,
        case 
            when salesOrderLineItem is not null 
                and salesOrderLineItem <> '' 
            then true else false 
        end as hasMatch : Boolean,

        toSoItemsByPo : Association to many V_SalesOrderItemsByPoId
            on toSoItemsByPo.po_id = poID,
        case
            when salesOrderLineItem is not null then 3
            when salesOrderLineItem is null then 2
            else 1
        end as matchingStatus_criticality : Integer
    };

    @UI.IsAIOperation
    action SyncDOX();
    
    @UI.IsAIOperation
    action generateMapping();
}

// annotate DocumentService with @requires :
// [
//     'authenticated-user'
// ];
