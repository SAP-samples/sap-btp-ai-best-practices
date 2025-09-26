import { MessageTimeAggregate, MessageTimeAggregates, Assessment, Situation, Situations } from "#cds-models/ReportService";
import { safeSendToAICore } from "./utils/AICore";
import { retry } from "./utils/Retry";
import { PriorityCriticalityMapping, TimeAggregateAssessmentResponse } from "./utils/types";
import { AggregateCriteria, MessageTimeAggregatesRepository } from "./repo/MessageTimeAggregatesRepository";

import cds from '@sap/cds';
import { AssessmentPromptBuilder } from "./utils/prompt/AssessmentPromptBuilder";
import { ReasoningStepsPromptBuilder } from "./utils/prompt/ReasoningStepsPromptBuilder";
import { ToolInvocations, Situations as _Situations } from "#cds-models/sap/btp/ai";
import { Tools } from "#cds-models/sap/btp/ai/tools";

const LOG = cds.log('code', { label: 'code' })

export class ReportService extends cds.ApplicationService {
    async init(): Promise<void> {
        //TODO input validation where? 
        await super.init();
        this.on("Test", this.test);
        this.on("assess", MessageTimeAggregates, req => {
            let ID: string;
            if(req.params.length == 1) {
                // page navigation like in log app #/MessageTimeAggregates('20250425-%252FXRP%252FENGINE_MSG-124')
                ID = req.params[0] as string;
            } else if(req.params.length == 2) {
                // page navigation like in situations app #/Situations(ID=190,IsActiveEntity=true)/timeAggregates('20250425-%252FXRP%252FENGINE_MSG-101')
                ID = req.params[1] as string;
            } else {
                LOG.error(`unexpected UI path parameters:: ${req.params}`);
                throw new Error("Unexpected action request parameters.");
            }
            this.assess(ID)
        });
        this.on("assessRelated", Situation, req => {
            const situation = req.params[0] as { ID: number, isActiveEntity: boolean }
            this.bulkAssess({ situationId: situation.ID })
        });
        this.on("bulkAssess", MessageTimeAggregates,
            req => this.bulkAssess({})
        );
        this.on("generateReasoningSteps", Situations,
            
            req => {
                const situation = req.params[0] as { ID: number, isActiveEntity: boolean }
                this.setReasoningSteps(situation.ID, true);
            }
        );
    }

    public assess = async (ID: unknown) => {

        if (typeof ID !== 'string') {
            throw new Error(`Invalid ID format :: ${ID}`);
          }
        const aggregateRepo = new MessageTimeAggregatesRepository(cds.tx())

        const aggregate: MessageTimeAggregate = await aggregateRepo.findOne(`${ID}`)
        if(!aggregate) {
            throw new Error(`Could not find aggregate with ID :: ${ID}`);
        }
        let situation: Situation = aggregate.situation!;
        // situation = await this.setReasoningSteps(situation, /*override*/ false)
        aggregate.situation = situation;
        await this.assessAndUpdateAggregate(aggregate)
    }

    public bulkAssess = async (criteria: AggregateCriteria) => {
        const aggregateRepo = new MessageTimeAggregatesRepository(cds.tx())
        const aggregates: MessageTimeAggregates = await aggregateRepo.find(criteria)

        if(aggregates.length == 0) {
            LOG.info("No aggregates for criteria :: ", criteria)
            return;
        }

        for (const aggregate of aggregates) {
            await this.assessAndUpdateAggregate(aggregate);
        }
        LOG.info("✅ Finished Assessments ✅")
        return aggregates;
    }

    public test = async (req: cds.Request) => {

        const template = `tell me a joke`

        // depending on model check the payload here https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core
        const payload = {
            messages: [{ role: "user", content: template }],
            max_completion_tokens: 16000,
        };

        const safeResponse = await safeSendToAICore<string>(payload)

        LOG.info(safeResponse);
    }

    private getAssessmentFromAI = async (prompt: string, aggregate: MessageTimeAggregate) => {
        
        const payload = {
            messages: [{ role: "user", content: prompt }],
            max_completion_tokens: 5000,
        };

        const safeResponse = await retry(3, () => safeSendToAICore<TimeAggregateAssessmentResponse>(payload));
        let result: Assessment;
        if (safeResponse.success) {

            result = {
                ID: cds.utils.uuid(),
                prompt: prompt,
                rawResponse: safeResponse.response["choices"][0]?.message?.content,
                summary: safeResponse.parsedData.summary,
                priority: safeResponse.parsedData.priority.code,
                priorityCriticality: PriorityCriticalityMapping[safeResponse.parsedData.priority.code],
                priorityReason: safeResponse.parsedData.priority.priorityReason,
                nextSteps: safeResponse.parsedData.nextSteps, 
                configurationChanges: safeResponse.parsedData.configurationChanges,
                rootCause: safeResponse.parsedData.rootCause,
                timeAggregate: aggregate
            }
        } else {
            result = {
                ID: cds.utils.uuid(),
                prompt: prompt,
                summary: 'The assessment could not be generated. Please try again.',
            }
        }
        LOG.info(`Assessment Successful.`)
        return result;
    }

    private async setReasoningSteps(situationID: number, override: boolean = false) {
        if(!situationID) {
            throw new Error("Associated situation with valid ID must be defined.");
        }
        //use db situation, not service projection
        const result: _Situations = await cds.db.run(SELECT.from(_Situations).where({ ID: situationID}))
        if(result.length != 1) {
            throw new Error("Associated situation with valid ID must be defined.");
        }
        const situation = result[0];
        if (override || !situation.reasoningSteps || situation.reasoningSteps.length === 0) {
            const reasoningSteps : string[] = await this._generateReasoningSteps(situation);
            situation.reasoningSteps = reasoningSteps;
            await cds.db.run(
                UPDATE('Situations')
                    .set({ reasoningSteps: reasoningSteps })
                    .where({ ID: situation.ID }));
        }
        LOG.info("✅ Finished Reasoning Steps ✅")
        return situation;
    }
    //generateReasoningSteps would overrule the .on declaration above with default behavior
    private async _generateReasoningSteps(situation: Situation) {
        const toolService = await cds.connect.to('ToolService');
        const tools: Tools = await toolService.run(SELECT.from('Tools'));

        LOG.info(`Generating Reasoning steps for situation :: ${situation.ID}`)
        const prompt = new ReasoningStepsPromptBuilder().situation(situation)
        .domainExpertComment(situation.comments!)
        .priorityDeterminationStrategy(situation.priorityDeterminationStrategy!)
        .tools(tools)
        .build();
        LOG.info(`🔥🔥🔥🔥 reasoning steps prompt :: ${prompt}`)
        const payload = {
            messages: [{ role: "user", content: prompt }],
            max_completion_tokens: 5000,
        };
        const safeResponse = await retry(3, () => safeSendToAICore<string[]>(payload));
        let result: string[] = [];
        if (safeResponse.success) {
            result = safeResponse.parsedData;
            LOG.info(`Success generating Reasoning steps for situation :: ${situation.ID}`)
        } else {
            LOG.warn(`Failed generating Reasoning steps for situation :: ${situation.ID}`)
        }
        return result;
    }

    private async assessAndUpdateAggregate(aggregate: MessageTimeAggregate) {
        LOG.info(`Assessing Aggregate :: ${aggregate.ID}`);
        // TODO create a placeholder assessment object to indicate to user that we're working on it. 

        const aggregateRepo = new MessageTimeAggregatesRepository(cds.tx())
        const history = await aggregateRepo.getMessageCountHistory(aggregate.situation!.ID as number, aggregate.date, 5)
        
        const toolService = await cds.connect.to('ToolService');
        const tools: Tools = await toolService.run(SELECT.from('Tools'));
        const toolInvocations: ToolInvocations = [];
        for(const tool of tools) {
            // @ts-ignore
            const toolOutput = await toolService.call(tool.name, JSON.stringify({'aggregateID':aggregate.ID}));
            toolInvocations.push({
                tool: tool,
                output: JSON.stringify(toolOutput)
            });
        }
        const prompt = new AssessmentPromptBuilder()
        .situation(aggregate.situation!)
        .priorityDeterminationStrategy(aggregate.situation!.priorityDeterminationStrategy!)
        .domainExpertComment(aggregate.situation!.comments!)
        .messageCountHistory(history)
        .toolInvocations(toolInvocations)
        .build();
        LOG.info(prompt)
        const aiAssessment: Assessment = await this.getAssessmentFromAI(prompt, aggregate);

        LOG.info(`Assessment completed :: `, aiAssessment);
        await cds.db.run(
            INSERT.into('Assessments').entries({
                ...aiAssessment,
                toolInvocations: toolInvocations
              }));
        await cds.db.run(
            UPDATE('MessageTimeAggregates')
                .set({ latestAssessment_ID: aiAssessment.ID })
                .where({ ID: aggregate.ID }));
        LOG.info(`Finished Assessment :: ${aiAssessment.ID} for Aggregate :: ${aggregate.ID}`);
    }
}