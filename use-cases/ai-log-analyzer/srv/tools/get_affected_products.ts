const cds = require('@sap/cds');

type InputSchema = {
    aggregateID: string
}


type ProductCountRow = {
    PRODUCT_ID: string;
    count: number;
    mean: number;
    stddev: number;
    min: number;
    max: number;
};

export async function getAffectedProducts(parameters: InputSchema) {
    const aggregateID = parameters.aggregateID;
    if(!aggregateID){
        throw new Error("Missing required paramater 'aggregateID' for tool: getAffectedProducts");
    }

    
    const query = `
        WITH counts AS (
            SELECT PRODUCT_ID, COUNT(*) AS "count"
            FROM "FF7D82B9D8284945AB85C0584FDB52E4"."SAP_BTP_AI_MESSAGES"
            WHERE timeAggregate_ID = ?
            GROUP BY PRODUCT_ID
        )

        SELECT
            PRODUCT_ID,
            "count",
            AVG("count") OVER () AS "mean",
            STDDEV_POP("count") OVER () AS "stddev",
            MIN("count") OVER () AS "min",
            MAX("count") OVER () AS "max"
        FROM counts
        ORDER BY "count" DESC
        `;
    const rawRows: ProductCountRow[] = await cds.db.run(query, [aggregateID]);
    if (!rawRows.length) {
        return {
          rows: [],
          stats: {
            mean: 0,
            stddev: 0,
            min: 0,
            max: 0,
            total: 0
          }
        };
      }
      
      const { mean, stddev, min, max } = rawRows[0];
      
      const rows = rawRows.map(({ PRODUCT_ID, count }) => ({ PRODUCT_ID, count }));

      return {
        rows,
        stats: {
          mean,
          stddev,
          min,
          max,
          total: rows.length
        }
    };
}