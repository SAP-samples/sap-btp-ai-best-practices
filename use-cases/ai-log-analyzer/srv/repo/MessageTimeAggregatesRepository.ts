// src/repositories/message-time-aggregates.ts
import cds, { Transaction } from '@sap/cds'
import { MessageTimeAggregate } from "#cds-models/ReportService";


export interface AggregateCriteria {
  ids?         : string[]        // one or many IDs
  situationId? : number          // FK to Situation
}

export class MessageTimeAggregatesRepository {
  constructor (private readonly tx: Transaction) {}

  async find (c: AggregateCriteria = {}) {
    const { ids, situationId } = c

    const q = SELECT
      .from (
        // @ts-ignore: implicit any in CQL callback
        `ReportService.MessageTimeAggregates`, (a => {
            a.ID, a.messageCount,
            a.date,
            a.situation (
              // @ts-ignore: implicit any in CQL callback
              s  => {
                s.ID,
                s.MSGNO,
                s.messageClass,
                s.messageShortText,
                s.messageType,
                s.businessArea,
                s.hlException,
                s.diagnosis,
                s.systemResponse,
                s.procedure,
                s.comments,
                s.referredMasterData,
                s.referredConfiguration,
                s.INCLUDE_messageClass,
                s.INCLUDE_messageShortText,
                s.INCLUDE_messageType,
                s.INCLUDE_businessArea,
                s.INCLUDE_hlException,
                s.INCLUDE_diagnosis,
                s.INCLUDE_systemResponse,
                s.INCLUDE_procedure,
                s.INCLUDE_comments,
                s.INCLUDE_referredMasterData,
                s.INCLUDE_referredConfiguration,
                s.priorityDeterminationStrategy,
                s.reasoningSteps
              }
            )
        })
      )


    // add filters only if supplied
    if (ids?.length) {
      q.where({ ID: { in: ids } })
    }  
    if (situationId != null) {
      q.where({ 'situation.ID': situationId })
    }
    return this.tx.run(q)
  }

  /** Convenience wrapper for exactly one aggregate (or `undefined`) */
  async findOne (id: string) {
    if (typeof id !== 'string') {
      throw new Error(`Invalid ID format :: ${id}`);
    }
    const rows = await this.find({ ids: [id] })
    return rows[0]
  }

  async getMessageCountHistory(situationId: number, date: unknown /* stupid CDS types or stupid me */, numberOfDays: number) {
    const q = SELECT
    .from (
      // @ts-ignore: implicit any in CQL callback
      `ReportService.MessageTimeAggregates`, (a => {
          a.date, a.messageCount
      })
    ).where({'situation.ID': situationId})
      .where(`
        situation_ID = ${situationId}
        and TO_DATE(date) BETWEEN ADD_DAYS(TO_DATE('${date}'), -${numberOfDays})
                      AND TO_DATE('${date}')
      `)
      .orderBy('date');
    
      return this.tx.run(q)
  }
}
