import { Situation } from "#cds-models/ReportService";
import { Priority, PriorityDeterminationStrategy } from "#cds-models/sap/btp/ai";
import { FixedPriorityStrategy, IPriorityDeterminationStrategy, PromptPriorityStrategy, TrendPriorityStrategy } from "../PriorityDeterminationStrategy";
import { SystemContext, MessageCountHistory } from "../types";
import cds from '@sap/cds';
import { BaseComposer } from "./BaseComposer";
import { Tools } from "#cds-models/sap/btp/ai/tools";

const LOG = cds.log('code', { label: 'code' })

const _agentIdentity: string = `You are an expert SAP Application Log Analyzer. Carefully review the Log Message Information. Your task is to define a step-by-step evaluation procedure that can be applied whenever a log message of type X is raised in the given business context. Do not evaluate a specific instance of the log. Instead, lay out the general logic to assess business impact, trend relevance, and criticality in future cases. Your output must be a plain-text list of numbered evaluation steps that can be followed consistently. First, select the specific tools you want to use to expand the context for this situation. Then list the reasoning steps to use all the resulting available context to find the root causes and next steps for the user. Always focus on how you can use the context to solve the problem.`

const _outputFormatExpectation: string = `**Required Output:** A JSON Array of strings, numbered chain of reasoning steps you will take to explain the business impact, observed data patterns, and interpretation logic. Example : ["Step 1: get_affected_products", "Step 2: review the affected products to see if there are outliers with especially high number of log messages to bring them to the users attention.", "Step 3: Formulate a situation summary that lists the highest priority items." ]`

const _contextMessageCountHistory = `You will be provided with a sample of recent message counts by date, here is a made up example: 
[{"date":"2025-04-21","messageCount":108406},{"date":"2025-04-22","messageCount":113532},{"date":"2025-04-23","messageCount":111841},{"date":"2025-04-24","messageCount":110276},{"date":"2025-04-25","messageCount":111555}]`

const _toolUsageDecision: string = `You have a set of tools you can call as a reasoning step to received additional context. You can use zero to all of the tools depending on your intuition if it is relevant to evaluate the semantic context and priority of the situation. To call a tool, add the "name" of the tool as a reasoning step in the list. Below is the list of available tools in JSON format: `;
export class ReasoningStepsPromptBuilder {
    _baseComposer: BaseComposer = new BaseComposer();
    _systemContext: SystemContext = {};
    _domainExpertComment: string = "";
    _tools: Tools = [];
    _priorityDeterminationStrategy: PriorityDeterminationStrategy = PriorityDeterminationStrategy.PROMPT;
    
    situation(situation: Situation) {
        this._systemContext = this._baseComposer.composeSystemInformationContext(situation);
        return this;
    }

    priorityDeterminationStrategy(strategy : PriorityDeterminationStrategy) {
        this._priorityDeterminationStrategy = strategy;
        return this;
    }

    domainExpertComment(comment:string) {
        this._domainExpertComment = comment; 
        return this;
    }

    tools(tools:Tools) {
        //only care about some fields
        this._tools = tools.map(({ name, description }) => ({ name, description }))
        return this;
    }


    build() {
        return `${this._baseComposer.getSystemInstructions()}

${_agentIdentity}

${_outputFormatExpectation}

**Inputs:**

**Log Message Information:**
${JSON.stringify(this._systemContext)}

**Tool context:**
${_toolUsageDecision}
${JSON.stringify(this._tools)}

**Message Count History:**
${_contextMessageCountHistory}

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
            case PriorityDeterminationStrategy.PROMPT:
                strategy = new PromptPriorityStrategy();
                break;
            default:
                LOG.error(`Unsupported priority strategy: ${strategyType}`)
                return;

        }
        
        return strategy.contextExample
        
    }

}
