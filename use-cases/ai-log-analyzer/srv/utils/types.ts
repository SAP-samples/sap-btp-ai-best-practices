import { Priority } from "#cds-models/sap/btp/ai";

export type PriorityCode = typeof Priority[keyof typeof Priority];

export type PriorityAssessment = {
    code: PriorityCode,
    priorityReason: string
}

export const PriorityCriticalityMapping: Record<PriorityCode, number> = {
    [Priority.LOW]: 3, // Neutral
    [Priority.MEDIUM]: 2, // Warning
    [Priority.HIGH]: 1, // Critical
} as const;

export type SafeResponse<T> = {
    /**
     * Whether API call to LLM was successful
     */
    success: true,
    /**
     * The full response returned by AI Core
     */
    response?: any, //TODO find correct type in AI core library
    /**
     * The first response, or a message to the end user if the call failed.
     */
    parsedData: T
} | {
    success: false,
};

export type TimeAggregateAssessmentResponse = {
    summary: string,
    priority: PriorityAssessment,
    nextSteps: string,
    configurationChanges: string,
    rootCause: string
}

export type SystemContext = {
    messageClass?: string,
    messageShortText?: string,
    messageType?: string,
    businessArea?: string,
    hlException?: string,
    diagnosis?: string,
    systemResponse?: string,
    procedure?: string,
    referredMasterData?: string,
    referredConfiguration?: string
}
export type MessageCountDay = {
    date: string,
    messageCount: number
}
export type MessageCountHistory = MessageCountDay[]
