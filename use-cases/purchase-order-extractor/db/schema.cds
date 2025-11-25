namespace com.sap.btp.ai;

using { managed, cuid } from '@sap/cds/common';

entity PurchaseOrders : managed, cuid {
  // Header Fields - flattened from headerFields array
  @Common.Label : 'Sender Postal Code'
  senderPostalCode      : String(20);
  @Common.Label : 'Sender State'
  senderState          : String(10);
  @Common.Label : 'Sender Street'
  senderStreet         : String(255);
  @Common.Label : 'Document Date'
  documentDate         : Date;
  @Common.Label : 'Document Number'
  documentNumber       : String(50);
  @Common.Label : 'Gross Amount'
  grossAmount          : Decimal(15,2);
  @Common.Label: 'Net Amount'
  netAmount            : Decimal(15,2);
  @Common.Label : 'Payment Terms'
  paymentTerms         : String(100);
  @Common.Label : 'Sender Address'
  senderAddress        : String(500);
  @Common.Label : 'Sender City'
  senderCity           : String(100);
  @Common.Label : 'Sender Country Code'
  senderCountryCode    : String(5);
  @Common.Label : 'Sender Fax'
  senderFax            : String(20);
  @Common.Label : 'Sender ID'
  senderId             : String(20);
  @Common.Label : 'Sender Name'
  senderName           : String(255);
  @Common.Label : 'Sender Phone'
  senderPhone          : String(20);
  @Common.Label : 'Ship To Address'
  shipToAddress        : String(500);
  @Common.Label : 'Ship To City'
  shipToCity           : String(100);
  @Common.Label : 'Ship To Country Code'
  shipToCountryCode    : String(5);
  @Common.Label : 'Ship To Fax'
  shipToFax            : String(20);
  @Common.Label : 'Ship To Name'
  shipToName           : String(255);
  @Common.Label : 'Ship To Phone'
  shipToPhone          : String(20);
  @Common.Label : 'Ship To Postal Code'
  shipToPostalCode     : String(20);
  @Common.Label : 'Ship To State'
  shipToState          : String(10);
  @Common.Label : 'Ship To Street'
  shipToStreet         : String(255);
  @Common.Label : 'Shipping Terms'
  shippingTerms        : String(100);
  
  // Processing status fields
  @Common.Label : 'Extraction Review Status'
  extractionReviewStatus : String(50) default 'not reviewed';
  @Common.Label : 'Payment Status'
  paymentStatus         : String(50) default 'unpaid';
  @Common.Label : 'Filename'
  filename             : String(255);
  @Common.Label : 'AI Extraction Review'
  aiExtractionReview   : String(5000);
  
  // Association to line items
  lineItems            : Composition of many LineItems on lineItems.purchaseOrder = $self;

  @Common.Label : 'Customer'
    customer             : Association to one Customers;
  @Common.Label : 'Customer Reason'
  customerReason       : String;

  salesOrders          : Association to many SalesOrders on $self.documentNumber = salesOrders.purchaseOrderNumber;
}

entity LineItems : managed, cuid {
  
  @Common.Label : 'Line Number in PDF'
  lineNumber : String(20);
  @Common.Label : 'Description'
  description              : String(500);
  @Common.Label : 'Net Amount'
  netAmount                : Decimal(15,2);
  @Common.Label : 'Quantity'
  quantity                 : Decimal(15,2);
  @Common.Label : 'Unit Price'
  unitPrice                : Decimal(15,2);
  @Common.Label : 'Supplier Material Number'
  supplierMaterialNumber   : String(50);
  @Common.Label : 'Customer Material Number'
  customerMaterialNumber   : String(50);
  
  // Association back to purchase order
  @Common.Label : 'Purchase Order'
  purchaseOrder            : Association to PurchaseOrders;

  @Common.Label : 'SAP Material Number'
  sapMaterialNumber        : String;
  @Common.Label : 'Material Selection Reason'
  materialSelectionReason  : String;

  mappingCandidates : Association to many LineItemCMIRCandidates on $self = mappingCandidates.lineitem;

  salesOrderItemMapping: Association to one LineItemSalesOrderItemCandidates  on $self = salesOrderItemMapping.purchaseOrderLineItem;
}

entity LineItemCMIRCandidates : cuid, managed {

  @Common.Label : 'Line Item'
  lineitem : Association to one LineItems;

  @Common.Label : 'Mapping'
  mapping : Association to one CMIRMappings; 
}

entity Customers : managed {

    @Common.Label : 'Customer Number'
    key ID : Integer;
    @Common.Label : 'Customer Name'
    name    : String;

    @Common.Label : 'Street'
    street : String(100);

    @Common.Label : 'Postal Code'
    postalCode : String(30);

    @Common.Label : 'City'
    city : String(100);

    @Common.Label : 'Region'
    region : String(3);

    purchaseOrders : Association to many PurchaseOrders on $self = purchaseOrders.customer;
}

entity CMIRMappings : cuid, managed {

    @Common.Label : 'Customer Material Number'
    customerMaterialNumber : String;

    @Common.Label : 'Supplier Material Number'
    supplierMaterialNumber : String;

    @Common.Label : 'Customer'
    customer : Association to one Customers;

    @Common.Label : 'Material Description'
    materialDescription : String;
}

entity SalesOrders : cuid, managed { 

  @Common.Label : 'Sales Order Number'
  salesOrderNumber : String;

  items: Composition of many SalesOrderItems on items.salesOrderNumber = $self.salesOrderNumber;

  @Common.Label : 'Purchase Order Number'
  purchaseOrderNumber : String;

  @Common.Label : 'Customer'
  customer : Association to one Customers;
  
  @Common.Label : 'Purchase Order'
  po : Association to one PurchaseOrders on po.documentNumber = $self.purchaseOrderNumber;
}

entity SalesOrderItems : cuid, managed {

    @Common.Label : 'Sales Order'
    salesOrder : Association to one SalesOrders on salesOrder.salesOrderNumber = $self.salesOrderNumber;

    @Common.Label : 'Sales Order Number'
    salesOrderNumber : String;

    @Common.Label : 'Line Number'
    lineNumber : Integer;

    @Common.Label : 'Date Created'
    dateCreated : Date;

    @Common.Label : 'SAP Material Number'
    sapMaterialNumber : String;

    @Common.Label : 'Sales Order Item Text'
    salesOrderItemText : String;

    @Common.Label : 'Customer Material Number'
    customerMaterialNumber : String;

    @Common.Label : 'Sold To'
    soldTo : String;
    
    @Common.Label : 'Ship To'
    shipTo : String;
}

entity LineItemSalesOrderItemCandidates : cuid, managed { 

    @assert.unique: { one_per_lineitem: [purchaseOrderLineItem] }
    @Common.Label : 'Purchase Order Line Item'
    purchaseOrderLineItem : Association to one LineItems;

    @Common.Label : 'Sales Order Line Item'
    salesOrderLineItem : Association to one SalesOrderItems;

    @Common.Label : 'Match Reason'
    reason : String;

     

}