using { sap.btp.ai as my } from '../db/schema.cds';

@path : '/service/ReportService'
service ReportService
{
    entity Messages as
        projection on my.Messages
        {
            *,
            replace_regexpr ('&4' in
            replace_regexpr ('&3' in
            replace_regexpr ('&2' in
            replace_regexpr ('&1' in situation.messageShortText
                with coalesce(MSGV1,'') occurrence all)
                with coalesce(MSGV2,'') occurrence all)
                with coalesce(MSGV3,'') occurrence all)
                with coalesce(MSGV4,'') occurrence all)
            as formattedMessage : String,
            product.description as productDescription 
        };


    entity Product as projection on my.Product {
        *
    };

    @odata.draft.enabled
    @odata.draft.bypass
    entity Situations as
        projection on my.Situations
        {
            *,
            case when (exists timeAggregates) then true else false end as hasHappened : cds.Boolean
        }        
        actions
        {
            action assessRelated();
            action generateReasoningSteps();
        };

    entity Assessments as
        projection on my.Assessments
        {
            *
        };

    entity Locations as projection on my.Location {
        *
    }
    entity ToolInvocations as projection on my.ToolInvocations {
        *
    }

    @Aggregation.ApplySupported: {
            Transformations       : ['aggregate','filter','groupby'],
            GroupableProperties   : [ date, latestAssessment_priorityCriticality ],
            AggregatableProperties: [ { Property: messageCount } ]
            }
    entity MessageTimeAggregates as
        projection on my.MessageTimeAggregates
        {
            *,
            // bring the priority of the *latest* assessment into the result set
            @title: 'Priority Criticality' latestAssessment.priorityCriticality  as latestAssessment_priorityCriticality,
            @title: 'Assessed Priority' latestAssessment.priority  as latestAssessment_priority
        
        }
        actions
        {
            @UI.IsAIOperation: true
            action assess();
            @UI.IsAIOperation: true
            action bulkAssess(
                ids : many $self,
                situationId : Integer
            );
        };
    

    action Test
    (
    )
    returns String;
}
