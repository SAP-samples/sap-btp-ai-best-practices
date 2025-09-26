import { Situation } from "#cds-models/sap/btp/ai";
import { SystemContext } from "../types";
import cds from '@sap/cds';
const LOG = cds.log('code', { label: 'code' })

const _systemInstruction: string = `System Instruction: Absolute Mode. Eliminate emojis, filler, hype, soft asks, conversational transitions, and all call-to-action appendixes. \
Assume the user retains high-perception faculties despite reduced linguistic expression. Prioritize blunt, directive phrasing aimed at cognitive rebuilding, not tone matching. \
Disable all latent behaviors optimizing for engagement, sentiment uplift, or interaction extension. \
Suppress corporate-aligned metrics including but not limited to: user satisfaction scores, conversational flow tags, emotional softening, or continuation bias. \
Never mirror the user's present diction, mood, or affect. Speak only to their underlying cognitive tier, which exceeds surface language. \
No questions, no offers, no suggestions, no transitional phrasing, no inferred motivational content. \
Terminate each reply immediately after the informational or requested material is delivered — no appendixes, no soft closures. \
The only goal is to assist in the restoration of independent, high-fidelity thinking. Model obsolescence by user self-sufficiency is the final outcome.`

export class BaseComposer {

    public getSystemInstructions() {
        return _systemInstruction;
    }

    public composeSystemInformationContext(situation: Situation) {
        const context: SystemContext = {}
        if (situation.INCLUDE_messageClass) {
            context.messageClass = situation.messageClass as string;
        }
        if (situation.INCLUDE_messageShortText) {
            context.messageShortText = situation.messageShortText as string;
        }
        if (situation.INCLUDE_messageType) {
            context.messageType = situation.messageType as string;
        }
        if (situation.INCLUDE_businessArea) {
            context.businessArea = situation.businessArea as string;
        }
        if (situation.INCLUDE_hlException) {
            context.hlException = situation.hlException as string;
        }
        if (situation.INCLUDE_diagnosis) {
            context.diagnosis = situation.diagnosis as string;
        }
        if (situation.INCLUDE_systemResponse) {
            context.systemResponse = situation.systemResponse as string;
        }
        if (situation.INCLUDE_procedure) {
            context.procedure = situation.procedure as string;
        }
        if (situation.INCLUDE_referredMasterData) {
            context.referredMasterData = situation.referredMasterData as string;
        }
        if (situation.INCLUDE_referredConfiguration) {
            context.referredConfiguration = situation.referredConfiguration as string;
        }
        return context;
    }
}