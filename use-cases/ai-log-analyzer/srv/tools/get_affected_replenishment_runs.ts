const cds = require('@sap/cds');

type InputSchema = {
    aggregateID: string
}

export async function getAffectedReplenishmentRuns(parameters: InputSchema) {
    const aggregateID = parameters.aggregateID;
    if(!aggregateID){
        throw new Error("Missing required paramater 'aggregateID' for tool: getAffectedReplenishmentRuns");
    }

    const query = cds.ql `SELECT from Messages {REPL_RUN_ID, count(REPL_RUN_ID) as count} where timeAggregate.ID = ${aggregateID} group by REPL_RUN_ID order by count desc`;
    const rows = await cds.db.run(query);
    return rows;
}