const cds = require('@sap/cds')

import { Customers, LineItem, LineItems, LineItemSalesOrderItemCandidates, SalesOrders } from "#cds-models/com/sap/btp/ai";
import { CMIRMappings, LineItemCMIRCandidates, CMIRMapping, LineItemCMIRCandidate, PurchaseOrders, PurchaseOrder, V_SalesOrderItemsByPoId, V_Materials, V_Material } from "#cds-models/DocumentService";
import { Readable } from "stream";
import { retry, safeSendToAICore } from "./utils/AICore";
import { executeHttpRequest } from '@sap-cloud-sdk/http-client';

import { Request } from '@sap/cds';

// Extend the Request interface to include custom properties
interface ExtendedRequest extends Request {
  _customerChanged?: boolean;
  _customerFrom?: string;
  _customerTo?: string;
}

const LOG = cds.log('code', 'info')

module.exports = cds.service.impl(async function(this: any) {
    function wants(req : Request, field : string) {
      const sel = req.query && req.query.SELECT
      if (!sel) return false                     // no $select → treat as "not requested"
      const cols = sel.columns || []
      return cols.some(c =>
        //@ts-ignore
        c === '*' ||
        (c.ref && c.ref.length === 1 && c.ref[0] === field)
      )
    }

    const { match } = V_SalesOrderItemsByPoId.actions;

    this.on(match, async (req: Request) => {

      const subject = await cds.db.run(SELECT.one.from(req.subject))
      LOG.info(`Manually matching ${JSON.stringify(subject)}`)
      
      const updated = await cds.db.run(UPDATE(LineItemSalesOrderItemCandidates)
          .set({salesOrderLineItem_ID: subject.so_item_id, reason: 'Manual Match'})
          .where({ID : req.params[0].ID}));
      LOG.info(`updated mappings:: ${JSON.stringify(updated)}`);
      
    })

    const {selectMaterial} = LineItemCMIRCandidate.actions;

    this.on(selectMaterial, async (req : Request) => {
      const selectedCandidate = await SELECT.one.from(req.subject)
      .columns(c => {
        c.lineitem((li: { ID: string; }) => { li.ID }), 
        c.mapping((m: CMIRMapping) => {m.supplierMaterialNumber})
      });
      await UPDATE(LineItems).byKey(selectedCandidate.lineitem.ID).set({'sapMaterialNumber': selectedCandidate.mapping.supplierMaterialNumber, materialSelectionReason: 'User match'})
    })

    const {matchSearchMaterial} = V_Material.actions;

    this.on(matchSearchMaterial, async (req : Request) => {
      const createCMIR = req.data.createCMIR;
      const selectedCandidate = await SELECT.one.from(req.subject)
      .columns(m => {
        m.supplierMaterialNumber, 
        m.materialDescription
      });
      await UPDATE(LineItems).byKey(req.params[1].ID).set({'sapMaterialNumber': selectedCandidate.supplierMaterialNumber, materialSelectionReason: 'User match based on Material Search'})

      if(createCMIR) {
        
        req.info({message: `Created CMIR record for ${selectedCandidate.supplierMaterialNumber}`})
      }
    })

    this.after('READ', PurchaseOrders, async (results: any, req: Request) => {
      if (!results) return;
      
      const purchaseOrders = Array.isArray(results) ? results : [results];
      
      for (const po of purchaseOrders) {
        if (po.ID) {
          // Get all line items for this purchase order
          const lineItems = await SELECT.from(LineItems).where({ purchaseOrder_ID: po.ID });
          
          // Calculate total count
          po.totalLineItemsCount = lineItems.length;
          
          // Calculate count with SAP material number
          po.lineItemsWithSapMaterialCount = lineItems.filter(item => 
            item.sapMaterialNumber && item.sapMaterialNumber.trim() !== ''
          ).length;
          
          // Calculate count without SAP material number
          po.lineItemsWithoutSapMaterialCount = lineItems.filter(item => 
            !item.sapMaterialNumber || item.sapMaterialNumber.trim() === ''
          ).length;

          //TODO add pdfContent from dox, only if the property is requested
          
          if(wants(req, 'pdfContent')) {
            const dox = await cds.connect.to('DOX-PREMIUM');
        
            try {

              const { data: ab } = await executeHttpRequest(
                { destinationName: 'DOX-PREMIUM' },
                { method: 'GET', url: `/document-information-extraction/v1/document/jobs/${po.ID}/file?clientId=${dox.options.CLIENT}`, responseType: 'arraybuffer' }
              );
              const buf = Buffer.from(ab);

              console.log('type=', typeof buf, 'isBuffer=', Buffer.isBuffer(buf),
              'ctor=', Object.prototype.toString.call(buf));
              po.pdfContent = Readable.from(buf);
            } catch (error) {
              LOG.error(`Error fetching PDF for PO ${po.ID}:`, error);
              po.pdfContent = null;
            }
          }
        }
      }
    });

    this.before(['UPDATE','PATCH'], PurchaseOrders, async (req : ExtendedRequest) => {
      if (!('customer_ID' in req.data)) return
      const { ID, customer_ID: to } = req.data
      if (!ID) return                       // not a single-row update
      const result = await SELECT.one`customer_ID`.from(PurchaseOrders).where({ ID })
      if (!result) return                   // purchase order not found
      const { customer_ID: from } = result
      req._customerChanged = from !== to
      req._customerFrom = from ? String(from) : undefined
      req._customerTo = to
    })

  this.after('UPDATE', PurchaseOrders, async ( po: PurchaseOrder, req: ExtendedRequest) => {
    const ID = po.ID
       // Get the updated purchase order to check if customer was set/changed
    const updatedPO = await cds.db.run(SELECT.one.from('PurchaseOrders').where({ID}));    
    if(req._customerChanged && updatedPO?.customer_ID && ID) {
      // If a customer has been selected or updated, set mapping candidates for line items
      await setMappingCandidatesForLineItems(ID, String(updatedPO.customer_ID));
    }


  });

  async function setMappingCandidatesForLineItems (purchaseOrder_ID : string, customer_ID: string) {
    LOG.info(`setMappingCandidatesForLineItems`)
    const lineItems = await SELECT.from(LineItems).columns('ID', 'supplierMaterialNumber', 'customerMaterialNumber', 'description').where({purchaseOrder_ID : purchaseOrder_ID});
    for await (let lineItem of lineItems) {
      await DELETE.from(LineItemCMIRCandidates).where({lineitem_ID : lineItem.ID})
      let supplierMaterialNumberMatches, customerMaterialNumberMatches, descriptionMatches;
      if (lineItem.supplierMaterialNumber) {
        LOG.info(`Looking for supplierMaterialNumber matches :: ${lineItem.supplierMaterialNumber}`)
        supplierMaterialNumberMatches = await SELECT.from(CMIRMappings).columns('*').where({customer_ID : customer_ID, supplierMaterialNumber : lineItem.supplierMaterialNumber})
        
        if(supplierMaterialNumberMatches.length == 1) {
          LOG.info(`found direct match :: ${JSON.stringify(lineItem)} => ${JSON.stringify(supplierMaterialNumberMatches[0].ID)}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': supplierMaterialNumberMatches[0].supplierMaterialNumber, materialSelectionReason: 'CMIR supplierMaterialNumber exact match'})
          await INSERT.into(LineItemCMIRCandidates).entries([{lineitem_ID : lineItem.ID, mapping_ID : supplierMaterialNumberMatches[0].ID}])
          continue;
        } else if(supplierMaterialNumberMatches.length > 1) {
          LOG.info(`Multiple supplierMaterialNumber match candidates ${JSON.stringify(supplierMaterialNumberMatches)}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': supplierMaterialNumberMatches[0].supplierMaterialNumber, materialSelectionReason: 'CMIR supplierMaterialNumber exact match - multiple - picked first'})
          await INSERT.into(LineItemCMIRCandidates).entries(supplierMaterialNumberMatches.map(m => {return {lineitem_ID : lineItem.ID, mapping_ID : m.ID}}))
          continue;
        }
      }
      if(lineItem.customerMaterialNumber) {
        const customerMaterial_to_sapMaterialMatches = await SELECT.from(CMIRMappings).columns('*').where({customer_ID : customer_ID, supplierMaterialNumber : lineItem.customerMaterialNumber})
        if(customerMaterial_to_sapMaterialMatches.length == 1) {
          LOG.info(`found direct match :: ${JSON.stringify(lineItem)} => ${JSON.stringify(customerMaterial_to_sapMaterialMatches[0])}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': customerMaterial_to_sapMaterialMatches[0].supplierMaterialNumber, materialSelectionReason: 'customerMaterialNumber -> sap Material exact match'})
          await INSERT.into(LineItemCMIRCandidates).entries({lineitem_ID : lineItem.ID, mapping_ID : customerMaterial_to_sapMaterialMatches[0].ID})
          continue;
        } else if(customerMaterial_to_sapMaterialMatches.length > 1) {
          LOG.info(`Setting customerMaterialNumber match candidates ${JSON.stringify(customerMaterial_to_sapMaterialMatches)}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': customerMaterial_to_sapMaterialMatches[0].supplierMaterialNumber, materialSelectionReason: 'customerMaterialNumber -> sap Material exact match - multiple - picked first'})
          await INSERT.into(LineItemCMIRCandidates).entries(customerMaterial_to_sapMaterialMatches.map(m => {return {lineitem_ID : lineItem.ID, mapping_ID : m.ID}}))
          continue;
        }
        
        LOG.info(`Looking for customerMaterialNumber matches :: ${lineItem.customerMaterialNumber}`)
        customerMaterialNumberMatches = await SELECT.from(CMIRMappings).columns('*').where({customer_ID : customer_ID, customerMaterialNumber : lineItem.customerMaterialNumber})
        
        if(customerMaterialNumberMatches.length == 1) {
          LOG.info(`found direct match :: ${JSON.stringify(lineItem)} => ${JSON.stringify(customerMaterialNumberMatches[0])}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': customerMaterialNumberMatches[0].supplierMaterialNumber, materialSelectionReason: 'CMIR customerMaterialNumber exact match'})
          await INSERT.into(LineItemCMIRCandidates).entries({lineitem_ID : lineItem.ID, mapping_ID : customerMaterialNumberMatches[0].ID})
          continue;
        } else if(customerMaterialNumberMatches.length > 1) {
          LOG.info(`Setting customerMaterialNumber match candidates ${JSON.stringify(customerMaterialNumberMatches)}`)
          await INSERT.into(LineItemCMIRCandidates).entries(customerMaterialNumberMatches.map(m => {return {lineitem_ID : lineItem.ID, mapping_ID : m.ID}}))
          continue;
        }

        LOG.info(`Looking for customerMaterialNumber matches in any customer :: ${lineItem.customerMaterialNumber}`)
        customerMaterialNumberMatches = await SELECT.from(CMIRMappings).columns('*').where({customerMaterialNumber : lineItem.customerMaterialNumber})
        
        if(customerMaterialNumberMatches.length == 1) {
          LOG.info(`found direct match :: ${JSON.stringify(lineItem)} => ${JSON.stringify(customerMaterialNumberMatches[0])}`)
          await UPDATE(LineItems).byKey(lineItem.ID).set({'sapMaterialNumber': customerMaterialNumberMatches[0].supplierMaterialNumber, materialSelectionReason: 'CMIR customerMaterialNumber any customer exact match'})
          await INSERT.into(LineItemCMIRCandidates).entries({lineitem_ID : lineItem.ID, mapping_ID : customerMaterialNumberMatches[0].ID})
          continue;
        } else if(customerMaterialNumberMatches.length > 1) {
          LOG.info(`Setting customerMaterialNumber match candidates ${JSON.stringify(customerMaterialNumberMatches)}`)
          await INSERT.into(LineItemCMIRCandidates).entries(customerMaterialNumberMatches.map(m => {return {lineitem_ID : lineItem.ID, mapping_ID : m.ID}}))
          continue;
        }
      }
      
    }
    
  }

  this.on('approve', async (req: Request) => {
    const ID = req.params[0].ID
    await cds.db.run(UPSERT.into('PurchaseOrders')
                      .entries({ID: ID, extractionReviewStatus: 'human reviewed'}))
  })

  this.on('review', async (req: Request) => {
    const ID = req.params[0].ID

    const purchaseOrder = await cds.db.run(SELECT.one.from('PurchaseOrders').where({ID : ID}));

    const previousHumanReviewedPOsForCustomer = await cds.db.run(
      SELECT.from('PurchaseOrders')
        .where({extractionReviewStatus: 'human reviewed', senderName : purchaseOrder.senderName })
        .orderBy('createdAt desc')
        .limit(3)
    )

    const template = `You are reviewing pdf extraction results for purchase orders. You are responsible to make sure the extracted data does not have extraction mistakes. To assist you in the assessment you are provided with up to the latest 3 documents that passed human review for this customer. You need to respond with a list of attributes to check and a concise note whether there is a risk of an extraction error for this purchase order. If you don't have context, don't make assumptions.
    
    ---

    previous purchase orders:
    
    ${JSON.stringify(previousHumanReviewedPOsForCustomer)}

    ---

    Purchase Order to be reviewed by you:

    ${JSON.stringify(purchaseOrder)}

    structure the response in JSON format: 
    {
      "assessment": <your assessment as string>
    }
    `

    // depending on model check the payload here https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core
    const payload = {
        messages: [{ role: "user", content: template }],
        max_completion_tokens: 16000,
    };

    const safeResponse = await safeSendToAICore<{assessment: String}>(payload)
    if(safeResponse.success) {
        const poAssessment : {assessment: String} = safeResponse.parsedData
        LOG.info(safeResponse.parsedData);
        
          const tx = cds.tx(req);
          await tx.run(
            UPSERT.into('PurchaseOrders')
                 .entries({ID: ID, aiExtractionReview: JSON.stringify(poAssessment.assessment), extractionReviewStatus: 'AI reviewed'})
          );
    }
    else {
      LOG.error(`❗❗❗❗❗❗❗ AI Core error for purchase order`)
    }
  })

  this.on('SyncDOX', async () => {
    const dox = await cds.connect.to('DOX-PREMIUM');
    
    const jobListResponse = await dox.send({
      query: `GET /document-information-extraction/v1/document/jobs?clientId=${dox.options.CLIENT}`,
    });

    const dbList = await cds.db.run(SELECT.from('PurchaseOrders'))
    const dbPOIDs = dbList.map((po: { ID: any; }) => po.ID)

    for(let extractedPO of jobListResponse.results) {
      
      if( !dbPOIDs.includes(extractedPO.id) ) {
        const jobResultResponse = await dox.send({
          query: `GET /document-information-extraction/v1/document/jobs/${extractedPO.id}?returnNullValues=true&clientId=${dox.options.CLIENT}`,
        });
        LOG.debug(JSON.stringify(jobResultResponse))

        // Helper function to find header field value by name
        const getHeaderFieldValue = (fieldName: string) => {
          const field = jobResultResponse.extraction.headerFields?.find((f: { name: string; }) => f.name === fieldName);
          return field ? field.value : null;
        };

        // Helper function to convert string to number or null
        const parseNumericValue = (value: any) => {
          if (value === null || value === undefined || value === '') {
            return null;
          }
          const parsed = parseFloat(value);
          return isNaN(parsed) ? null : parsed;
        };

        // Extract header fields from the new schema format
        const headerData = {
          senderPostalCode: getHeaderFieldValue('senderPostalCode'),
          senderState: getHeaderFieldValue('senderState'),
          senderStreet: getHeaderFieldValue('senderStreet'),
          documentDate: getHeaderFieldValue('documentDate'),
          documentNumber: getHeaderFieldValue('documentNumber'),
          grossAmount: parseNumericValue(getHeaderFieldValue('grossAmount')),
          netAmount: parseNumericValue(getHeaderFieldValue('netAmount')),
          paymentTerms: getHeaderFieldValue('paymentTerms'),
          senderAddress: getHeaderFieldValue('senderAddress'),
          senderCity: getHeaderFieldValue('senderCity'),
          senderCountryCode: getHeaderFieldValue('senderCountryCode'),
          senderFax: getHeaderFieldValue('senderFax'),
          senderId: getHeaderFieldValue('senderId'),
          senderName: getHeaderFieldValue('senderName'),
          senderPhone: getHeaderFieldValue('senderPhone'),
          shipToAddress: getHeaderFieldValue('shipToAddress'),
          shipToCity: getHeaderFieldValue('shipToCity'),
          shipToCountryCode: getHeaderFieldValue('shipToCountryCode'),
          shipToFax: getHeaderFieldValue('shipToFax'),
          shipToName: getHeaderFieldValue('shipToName'),
          shipToPhone: getHeaderFieldValue('shipToPhone'),
          shipToPostalCode: getHeaderFieldValue('shipToPostalCode'),
          shipToState: getHeaderFieldValue('shipToState'),
          shipToStreet: getHeaderFieldValue('shipToStreet'),
          shippingTerms: getHeaderFieldValue('shippingTerms')
        };

        const filename = jobResultResponse.fileName || 'unknown';


        // Insert Purchase Order
        try {
          const poEntry = {
            ID: extractedPO.id,
            extractionReviewStatus: 'not reviewed',
            paymentStatus: 'unpaid',
            filename: filename,
            sender_name: headerData.senderName, // Set up the association key
            customer_ID: null as string | null,
            customerReason: null as string | null,
            ...headerData
          };

          const {customerId, reason} = await findCustomerByContext(JSON.stringify(poEntry));
          poEntry.customer_ID = customerId;
          poEntry.customerReason = reason;

          const insertPOResult = await cds.db.run(INSERT.into('PurchaseOrders').entries(poEntry));
          LOG.info(`insert purchase order result :: ${JSON.stringify(insertPOResult)}`)

          // Process line items if they exist
          if (jobResultResponse.extraction.lineItems && Array.isArray(jobResultResponse.extraction.lineItems)) {
            for (let lineItemFields of jobResultResponse.extraction.lineItems) {
              // Helper function to find line item field value by name
              const getLineItemFieldValue = (fieldName: string) => {
                const field = lineItemFields.find((f: { name: string; }) => f.name === fieldName);
                return field ? field.value : null;
              };

              const lineItemData = {
                lineNumber: getLineItemFieldValue('itemNumber'),
                description: getLineItemFieldValue('description'),
                netAmount: getLineItemFieldValue('netAmount'),
                quantity: getLineItemFieldValue('quantity'),
                unitPrice: getLineItemFieldValue('unitPrice'),
                supplierMaterialNumber: getLineItemFieldValue('supplierMaterialNumber'),
                customerMaterialNumber: getLineItemFieldValue('customerMaterialNumber'),
                purchaseOrder_ID: extractedPO.id
              };

              try {
                const insertLineItemResult = await cds.db.run(INSERT.into('LineItems').entries(lineItemData));
                LOG.info(`insert line item result :: ${JSON.stringify(insertLineItemResult)}`)
              } catch (e) {
                LOG.error("Error when persisting line item", e)
              }
            }
          }
          if(poEntry.customer_ID != null) {
            // If a customer has been found, set mapping candidates for line items
            await setMappingCandidatesForLineItems(poEntry.ID, String(poEntry.customer_ID));
          }

        } catch (e) {
          LOG.error("Error when persisting new purchase order", e)
        }
      }
    }
    return `sync completed`;
  })

this.on('generateMapping', async () => {

    const poList = await cds.db.run(SELECT.from(PurchaseOrders, p => {
      p.ID, p.documentNumber, p.documentDate, p.grossAmount, p.netAmount, p.paymentTerms
      p.senderName, p.senderAddress, p.senderCity, p.senderState, p.senderPostalCode, p.senderCountryCode
      p.shipToName, p.shipToAddress, p.shipToCity, p.shipToState, p.shipToPostalCode, p.shipToCountryCode
      p.customer(c => { c.ID, c.name })
      p.lineItems(li => {
        li.ID, li.lineNumber, li.description, li.quantity, li.unitPrice, li.netAmount
        li.supplierMaterialNumber, li.customerMaterialNumber
      })
      p.salesOrders(so => {
        so.salesOrderNumber, so.purchaseOrderNumber
        so.items(i => {
          i.ID, i.salesOrderNumber, i.lineNumber, i.dateCreated
          i.sapMaterialNumber, i.salesOrderItemText, i.soldTo, i.shipTo
        })
      })
    }))

    for(let extractedPO of poList) {
      try {
      const template = `You match purchase-order line items to the best sales-order line items from the provided JSON only. Do not invent data.

# Task
For each PO line item, pick at most one best matching SO item from the linked sales orders. If no good match exists, return null for that PO line. Provide a short reason per decision.

# Data
${JSON.stringify(extractedPO)}

# Fields to use
- PO line item: ID, lineNumber, description, quantity, unitPrice, netAmount, supplierMaterialNumber, customerMaterialNumber.
- SO item: ID, salesOrderNumber, lineNumber, dateCreated, sapMaterialNumber, salesOrderItemText, soldTo, shipTo.
- PO header: shipToName, shipToAddress et al. Use for consistency checks only.

# Evidence priority
Rank candidates using these signals in order. Break ties with the next signal.
1) Material-number equality:
   - Prefer PO.supplierMaterialNumber == SO.sapMaterialNumber.
   - Else PO.customerMaterialNumber == SO.sapMaterialNumber.
   - There MAY be up to 2 letters trailing the customerMaterialNumber in the SO.sapMaterialNumber, match in this case.
2) Matching line numbers:
   - Compare PO.lineNumber and SO.lineNumber. Matching lineNumbers are a strong signal.
   - if there are multiple SO items with the same lineNumber, this signal cannot be used. 
3) Text similarity:
   - Normalize text (lowercase, strip punctuation and common stopwords).
   - Compare PO.description to SO.salesOrderItemText. Prefer higher semantic similarity and shared rare tokens.
4) Quantity and price sanity:
   - Favor quantities within ±10%.
   - If unitPrice present, check netAmount ≈ quantity × unitPrice. Prefer SO items consistent with that magnitude.
5) Ship-to and customer context:
   - Prefer SO items whose soldTo/shipTo align with PO header ship-to signals.
6) Recency:
   - If still tied, prefer the most recent SO item by dateCreated.

# Abstain rule
If the top candidate lacks any material-number equality
AND text similarity is low
AND quantity/price sanity fails,
then set salesOrderItemId to null and explain “no sufficient evidence.”

# One-to-one constraint
Each PO line maps to at most one SO item. Multiple PO lines may map to the same SO item only if each has independent strong evidence.

# Output format (JSON only)
Return a compact JSON array, no prose. Each element:
{
  "purchaseOrderLineItemId": "<PO line item ID>",
  "salesOrderItemId": "<SO item ID or null>",
  "reason": "<one sentence, 15–30 words>"
}

# Procedure
1) List PO line items in input order.
2) Build a shortlist per PO line using material-number equality and high text similarity.
3) Rank with the evidence priority.
4) Apply the abstain rule.
5) Emit the JSON array. Use IDs exactly as given. No extra fields.

# Example output shape
[
  {"purchaseOrderLineItemId":"f0b1...","salesOrderItemId":"a9c3...","reason":"Supplier material equals SO sapMaterial; quantities within 5%; descriptions closely aligned."},
  {"purchaseOrderLineItemId":"f0b2...","salesOrderItemId":null,"reason":"No material match, low text similarity, and quantity mismatch >30%."}
]

# Validation
- One object per PO line item.
- Only IDs present in the input JSON.
- Null allowed when abstaining.
- Preserve the order of PO line items.`

    // depending on model check the payload here https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core
    const payload = {
        messages: [{ role: "user", content: template }],
        max_completion_tokens: 50000,
    };

    const safeResponse = await retry(3, () => safeSendToAICore<[{purchaseOrderLineItemId: string | null, salesOrderItemId: string | null, reason: string}]>(payload))
    if(safeResponse.success) {
      const response : [{purchaseOrderLineItemId: string | null, salesOrderItemId: string | null, reason: string}] = safeResponse.parsedData

      const inserted = await cds.db.run(INSERT.into(LineItemSalesOrderItemCandidates).entries(safeResponse.parsedData.map(r => {
        return { purchaseOrderLineItem_ID: r.purchaseOrderLineItemId, salesOrderLineItem_ID: r.salesOrderItemId, reason: r.reason}
      })));
      LOG.info(`inserted mappings:: ${JSON.stringify(inserted)}`);
    }
    else {
      LOG.error(`❗❗❗❗❗❗❗ AI Core error for purchase order`)
    }
    } catch (e) {
      LOG.info(`caught unexpected exception`, e)
  }
  } 
      
    return `matching completed`;
  })


  async function findCustomerByContext(po: string) : Promise<{ customerId: string | null; reason: string; }> {

    const customers = await cds.db.run(
      SELECT.from(Customers).columns('ID', 'name', 'street', 'city', 'postalCode', 'region')
    )

    const template = `You are reviewing pdf extraction results for purchase orders of a manufacturing company. You are responsible to find the right customer ID for the uploaded document. To assist you in the assessment you are provided with the extracted data of the purchase order, the file name and the list of existing customers and their IDs.  If you don't have enough context to make an informed decision, don't make assumptions, don't select a customer and return null. Add a short sentence on why the selection was made in the reason property.
    
    ---

    customers:
    
    ${JSON.stringify(customers)}

    ---

    Purchase Order to be assigned a customer:

    ${JSON.stringify(po)}

    structure the response in JSON format: 
    {
      "customerId": string || null,
      "reason": string
    }
    `

    // depending on model check the payload here https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core
    const payload = {
        messages: [{ role: "user", content: template }],
        max_completion_tokens: 16000,
    };

    const safeResponse = await safeSendToAICore<{customerId: string | null, reason: string}>(payload)
    if(safeResponse.success) {
        const response : {customerId: string | null, reason: string} = safeResponse.parsedData
        LOG.info(JSON.stringify(safeResponse.parsedData));
        return response;
    }
    else {
      LOG.error(`❗❗❗❗❗❗❗ AI Core error for purchase order`)
      return {customerId: null, reason: `There was a problem detecting the customer`};
    }
  }

})
