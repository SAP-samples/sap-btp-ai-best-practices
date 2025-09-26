namespace sap.btp.ai;

using { managed, cuid } from '@sap/cds/common';

using { sap.btp.ai.tools as tools } from '../db/tools-schema.cds';
 

type PriorityDeterminationStrategy: String enum {
    LOW = 'L';
    MEDIUM = 'M';
    HIGH = 'H';
    PROMPT = 'P';
    TREND = 'T';
}

type Priority: String enum {
    LOW = 'low';
    MEDIUM = 'medium';
    HIGH = 'high';
}

entity Situations : managed
{
    key ID : Integer;
    MSGNO : String;
    messageClass : String;
    messageShortText : String;
    messageType : String;
    businessArea : String;
    hlException : String;
    diagnosis : String;
    systemResponse : String;
    procedure : String;
    comments : String;
    referredMasterData : String;
    referredConfiguration : String;
    messages : Association to many Messages on $self.MSGNO = messages.MSGNO and $self.messageClass = messages.MSGID;
    INCLUDE_messageClass : Boolean;
    INCLUDE_messageShortText : Boolean;
    INCLUDE_messageType : Boolean;
    INCLUDE_businessArea : Boolean;
    INCLUDE_hlException : Boolean;
    INCLUDE_diagnosis : Boolean;
    INCLUDE_systemResponse : Boolean;
    INCLUDE_procedure : Boolean;
    INCLUDE_comments : Boolean;
    INCLUDE_referredMasterData : Boolean;
    INCLUDE_referredConfiguration : Boolean;
    timeAggregates: Association to many MessageTimeAggregates on $self = timeAggregates.situation;
    priorityDeterminationStrategy: PriorityDeterminationStrategy default #PROMPT;
    reasoningSteps: many String;
}

entity Messages : managed
{
    key ID : Integer;
    MANDT : String;
    MSGHDL : String;
    PROBCLASS : String;
    MSGTYPE : String;
    MSGID : String;
    MSGNO : String;
    MSGV1 : String;
    MSGV2 : String;
    MSGV3 : String;
    MSGV4 : String;
    IS_ASSIGNED : String;
    CREATIONTSTAMP : String;
    USR : String;
    TCODE : String;
    PROG : String;
    MOD : String;
    DELETIONTSTAMP : String;
    STATUS : String;
    DELETION_INDICATOR : String;
    RELCONT : String;
    MSGPSTAMP : String;
    RUNDATE : String;
    PRODUCT_ID : String;
    product: Association to one Product on $self.PRODUCT_ID = product.ID; 
    LOCATION_ID : String;
    location: Association to one Location on $self.LOCATION_ID = location.ID; 
    LOCATIONFROM_ID : String;
    locationFrom: Association to one Location on $self.LOCATIONFROM_ID = locationFrom.ID; 
    PRODUCT_HIERARCHY_ID : String;
    LOCATION_HIERARCHY_ID : String;
    OFFER_ID : String;
    VENDOR_FUND_ID : String;
    KPRM : String;
    MODEL_ID : String;
    FORECAST_ID : String;
    CONTROLLER_ID : String;
    JOB_ID : String;
    DIAG_ID : String;
    SALES_ORG_ID : String;
    DISTR_CHNL_ID : String;
    ORDER_CHNL_ID : String;
    INBOUND_CONTEXT : String;
    COUNTER : String;
    ALLO_PLAN_ID : String;
    COLOR : String;
    MARKET_UNIT_ID : String;
    ALLO_WL_ID : String;
    REFERENCE_ID : String;
    ORDERPLANITEMUUID : String;
    REPL_RUN_ID : String;
    ORDERPLANITEM4AUUID : String;
    situation : Association to one Situations on situation.MSGNO = $self.MSGNO and situation.messageClass = $self.MSGID;
    timeAggregate : Association to one MessageTimeAggregates;
}

entity MessageTimeAggregates : managed
{
    key ID : String;
    timeFrom : Timestamp;
    timeTo : Timestamp;
    date : Date;
    @mandatory
    situation : Association to one Situations;
    messageCount : Integer;
    assessment: Association to many Assessments on $self = assessment.timeAggregate;
    messages : Association to many Messages on $self = messages.timeAggregate;
    latestAssessment : Association to one Assessments;
}

entity Assessments : managed
{
    key ID : UUID;
    prompt: LargeString;
    rawResponse: LargeString;
    summary: String;
    priority: Priority;
    priorityCriticality: Integer;
    priorityReason: String;
    nextSteps: String;
    configurationChanges: String;
    rootCause: String;
    timeAggregate: Association to one MessageTimeAggregates;
    toolInvocations: Composition of many ToolInvocations on toolInvocations.assessment = $self;
}

entity ToolInvocations : managed, cuid {

    tool: Association to one tools.Tools;
    // Parse with tool.outputSchema. Validated at tool execution. 
    output: LargeString;
    assessment: Association to one Assessments;
}

entity Product {
    key ID: String;
    messages: Association to many Messages on messages.PRODUCT_ID = $self.ID;
    description: String;
}

entity Location {
    key ID: String;
    name: String;
    location_messages: Association to many Messages on location_messages.LOCATION_ID = $self.ID;
    locationFrom_messages: Association to many Messages on locationFrom_messages.LOCATIONFROM_ID = $self.ID;
}