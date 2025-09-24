const cds = require('@sap/cds');

type InputSchema = {
    aggregateID: string
}

type WarehouseCountRow = {
    LOCATIONFROM_ID: string;
    count: number;
    mean: number;
    stddev: number;
    min: number;
    max: number;
  };

export async function getAffectedWarehouses(parameters: InputSchema) {
    const aggregateID = parameters.aggregateID;
    if(!aggregateID){
        throw new Error("Missing required paramater 'aggregateID' for tool: getAffectedWarehouses");
    }

    const query = `
        WITH counts AS (
            SELECT LOCATIONFROM_ID, COUNT(*) AS "count"
            FROM "FF7D82B9D8284945AB85C0584FDB52E4"."SAP_BTP_AI_MESSAGES"
            WHERE timeAggregate_ID = ?
            GROUP BY LOCATIONFROM_ID
        )

        SELECT
            LOCATIONFROM_ID,
            "count",
            AVG("count") OVER () AS "mean",
            STDDEV_POP("count") OVER () AS "stddev",
            MIN("count") OVER () AS "min",
            MAX("count") OVER () AS "max"
        FROM counts
        ORDER BY "count" DESC
        `;
    const rawRows: WarehouseCountRow[] = await cds.db.run(query, [aggregateID]);
    
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
      
      const rows = rawRows.map(({ LOCATIONFROM_ID, count }) => ({ LOCATIONFROM_ID, count }));

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