import { Situation } from "#cds-models/ReportService";
import { Priority, PriorityDeterminationStrategy, ToolInvocations } from "#cds-models/sap/btp/ai";
import { FixedPriorityStrategy, IPriorityDeterminationStrategy, TrendPriorityStrategy } from "../PriorityDeterminationStrategy";
import { SystemContext, MessageCountHistory } from "../types";
import cds from '@sap/cds';
import { BaseComposer } from "./BaseComposer";

const LOG = cds.log('code', { label: 'code' })

const _agentIdentity: string = `You are an expert SAP Application Log Analyzer with extensive experience. You have been provided with aggregated log messages for a specific error type and a domain expert comment (if available). Using this data, your task is to provide an initial assessment of an issue in the application log. You want to prepare the reviewer as best as possible to act on the situation.`

const _outputFormatExpectation: string = `
**Important:** Output your response strictly in a valid JSON format using the keys listed below. Do not include any additional text or commentary outside of the JSON structure.

**Required JSON Format:**

{
"summary": "<Provide a summary here>",
"priority": {
    "code": "<Indicate the priority level as one of (low,medium,high)>",
    "priorityReason": "<Indicate the reason for choosing this priority in 1 sentence or less>"
},
"nextSteps": "<List of bullet points for the recommended next steps as string>",
"configurationChanges": "<Suggest possible configuration changes as string>",
"rootCause": "<State the likely root cause if you know it>"
}`

export class AssessmentPromptBuilder {
    _baseComposer: BaseComposer = new BaseComposer();
    _systemContext: SystemContext = {};
    _domainExpertComment: string = "";
    _messageCountHistory: MessageCountHistory = [];
    _priorityDeterminationStrategy: PriorityDeterminationStrategy = PriorityDeterminationStrategy.PROMPT;
    _toolInvocations: ToolInvocations = [];
    
    situation(situation: Situation) {
        this._systemContext = this._baseComposer.composeSystemInformationContext(situation);
        return this;
    }

    messageCountHistory(history: MessageCountHistory) {
        this._messageCountHistory = history;
        return this;
    }

    priorityDeterminationStrategy(strategy : PriorityDeterminationStrategy) {
        this._priorityDeterminationStrategy = strategy;
        return this;
    }

    toolInvocations(toolInvocations : ToolInvocations) {
        this._toolInvocations = toolInvocations;
        return this;
    }

    domainExpertComment(comment:string) {
        this._domainExpertComment = comment; 
        return this;
    }


    build() {
        const usePromptForPriority: boolean = this._priorityDeterminationStrategy == PriorityDeterminationStrategy.PROMPT

        return `${this._baseComposer.getSystemInstructions()}

${_agentIdentity}

${this._composeExpectation(usePromptForPriority)}

${usePromptForPriority ? /* skip */'' : '**Priority Assessment:**'}
${usePromptForPriority ? /* skip */'' : JSON.stringify(this._composePriorityContext(this._priorityDeterminationStrategy))}

${_outputFormatExpectation}

**Log Message Information:**
${JSON.stringify(this._systemContext)}

**Message Count History:**
${JSON.stringify(this._messageCountHistory)}


**Data context:**
${this._composeDataContext(this._toolInvocations)}

**Domain Expert Comment:**

{
"comments": "${this._domainExpertComment}"
}
`
    }

    private _composePriorityContext(strategyType: PriorityDeterminationStrategy) {
        let strategy: IPriorityDeterminationStrategy;
        LOG.info(`Determine Priority using :: ${strategyType}`)
        switch(strategyType){
            case PriorityDeterminationStrategy.LOW:
                strategy = new FixedPriorityStrategy(Priority.LOW)
                break;
            case PriorityDeterminationStrategy.MEDIUM:
                strategy = new FixedPriorityStrategy(Priority.MEDIUM)
                break;
            case PriorityDeterminationStrategy.HIGH:
                strategy = new FixedPriorityStrategy(Priority.HIGH)
                break;
            case PriorityDeterminationStrategy.TREND:
                strategy = new TrendPriorityStrategy()
                break;
            default:
                LOG.error(`Unsupported priority strategy: ${strategyType}`)
                return;

        }
        
        return strategy.prioritize(this._messageCountHistory)
        
    }

    private _composeDataContext(toolInvocations: ToolInvocations) {
        try {
            let result: string = 
`We decided to load the following data analytics about the affected messages. Please review them and consider them in the assessment if the data is relevant to the root cause or next steps.
`

            for(let i = 0; i < toolInvocations.length; i++) {
                const output = JSON.parse(toolInvocations[i].output!)
                if(toolInvocations[i].tool!.name === 'get_affected_replenishment_runs') {
                    result += 
`${i}. Tool used to retrieve data context ${i}: ${toolInvocations[i].tool?.description} 
- message count by replenishment runs: ${JSON.stringify(output)}
`           
                    continue;
                } else {
                    result += 
`${i}. Tool used to retrieve data context ${i}: ${toolInvocations[i].tool?.description} 
-  top 5 result rows by count: ${JSON.stringify(output.rows.slice(0,5))}
-  stats: ${JSON.stringify(output.stats)}
`
                }
            }

            return result;
        } catch (e) {
            LOG.warn("Error when parsing tool data context")
            LOG.warn(e)
            return "No data available"
        }
    }

    _composeExpectation(usePromptForPriority:boolean) {
        return `
        In your analysis, please address the following areas:

        - **Summary:** Begin with a concise overview of your assessment.
        - **Priority:** ${usePromptForPriority ? 'Evaluate and state the priority level of the issue (high, medium, low).' : 'return the priority value from the priority context'}
        - **Next Steps:** Outline clear recommendations for further investigation or remedial actions as bullet points.
        - **Possible Configuration Changes:** Suggest any configuration modifications in the System that could help mitigate the issue or reduce noisy logs. 
        - **Root Cause:** Identify the likely root cause behind the issue.
        `
    }
}
