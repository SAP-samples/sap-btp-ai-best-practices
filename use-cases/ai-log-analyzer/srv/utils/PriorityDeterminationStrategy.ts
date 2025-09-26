import { Priority } from "#cds-models/sap/btp/ai";
import { MessageCountHistory, PriorityAssessment } from "./types"

export interface IPriorityDeterminationStrategy { 
    /**
     * An explanation for the LLM on what to expect at runtime. 
     */
    contextExample: String;
    prioritize(history?: MessageCountHistory) : PriorityAssessment;
}

export class FixedPriorityStrategy implements IPriorityDeterminationStrategy {
  contextExample = `You will be provided with a fixed priority asssessment based on User input. Made up Example:
{"code":"low","priorityReason":"Fixed priority set to 'low'."}
`

    constructor(private readonly fixed: Priority) {}

  
    prioritize(_history?: MessageCountHistory): PriorityAssessment {
      return { code: this.fixed, priorityReason: `Fixed priority set to '${this.fixed}'.` };
    }
}

export class PromptPriorityStrategy implements IPriorityDeterminationStrategy {
    contextExample = `You will be assessing priority of the situation based on Context provided.`
    
    constructor() {}
      
    prioritize(_history?: MessageCountHistory): PriorityAssessment {
      throw new Error("Prompt prioritization happens in the prompt.")
    }
}


export class TrendPriorityStrategy implements IPriorityDeterminationStrategy {
    contextExample = `You will be provided with a priority asssessment based on Trend analysis. Made up Example:
    {"code":"low","priorityReason":"Latest count 111555 is only 0.5% above the average (111014), within the 10% threshold."}
    `
    constructor(
      private readonly varianceThresholdMedium: number = 10.0,
      private readonly varianceThresholdHigh:   number = 20.0
    ) {}
  
    prioritize(history: MessageCountHistory = []): PriorityAssessment {
      const count = history.length;
      // 1) Not enough data to form a trend
      if (count < 2) {
        return {
          code: Priority.LOW,
          priorityReason:
            count === 0
              ? "No history available; defaulting to LOW priority."
              : "Only one data point available; insufficient history for trend analysis. Defaulting to LOW priority.",
        };
      }
  
      // 2) Ensure correct ordering by ISO‐date
      const sorted = [...history].sort((a, b) => a.date.localeCompare(b.date));
      const latest = sorted[sorted.length - 1];
      const previous = sorted.slice(0, sorted.length - 1);
  
      // 3) Compute average of all but the latest
      const sum = previous.reduce((acc, e) => acc + e.messageCount, 0);
      const avg = sum / previous.length;
  
      // 4) Calculate percent‐change
      const diff = latest.messageCount - avg;
      const percentChange = (diff / avg) * 100;
      const absChange = Math.abs(percentChange);
  
      // 5) Determine priority
      let code: Priority;
      let reason: string;
  
      if (absChange >= this.varianceThresholdHigh) {
        code = Priority.HIGH;
        reason = `Latest count ${latest.messageCount} is ${percentChange.toFixed(
          1
        )}% ${percentChange > 0 ? "above" : "below"} the ${previous.length}-day average (${avg.toFixed(
          0
        )}), exceeding the ${this.varianceThresholdHigh}% HIGH threshold.`;
      } else if (absChange >= this.varianceThresholdMedium) {
        code = Priority.MEDIUM;
        reason = `Latest count ${latest.messageCount} is ${percentChange.toFixed(
          1
        )}% ${percentChange > 0 ? "above" : "below"} the average (${avg.toFixed(
          0
        )}), exceeding the ${this.varianceThresholdMedium}% MEDIUM threshold.`;
      } else {
        code = Priority.LOW;
        reason = `Latest count ${latest.messageCount} is only ${percentChange.toFixed(
          1
        )}% ${percentChange > 0 ? "above" : "below"} the average (${avg.toFixed(
          0
        )}), within the ${this.varianceThresholdMedium}% threshold.`;
      }
  
      return {
        code,
        priorityReason: reason,
      };
    }
}