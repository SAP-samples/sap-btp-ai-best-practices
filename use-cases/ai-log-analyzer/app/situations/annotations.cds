using ReportService as service from '../../srv/report-service';

annotate service.Situations with {
    MSGNO @Common.Label : 'Message Number'
};

annotate service.Situations with {
    messageClass @Common.Label : 'Message Class'
};
annotate service.Situations with {
    priorityDeterminationStrategy @Common.Label : 'Priority Determination Strategy'
};
annotate service.Situations with {
    hasHappened @Common.Label : 'Has Happened'
};
annotate service.Situations with @(
    UI.FieldGroup #GeneratedGroup : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_messageClass',
                Value : INCLUDE_messageClass,
            },
            {
                $Type : 'UI.DataField',
                Label : 'messageClass',
                Value : messageClass,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_messageShortText',
                Value : INCLUDE_messageShortText,
            },
            {
                $Type : 'UI.DataField',
                Label : 'messageShortText',
                Value : messageShortText,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_messageType',
                Value : INCLUDE_messageType,
            },
            {
                $Type : 'UI.DataField',
                Label : 'messageType',
                Value : messageType,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_businessArea',
                Value : INCLUDE_businessArea,
            },
            {
                $Type : 'UI.DataField',
                Label : 'businessArea',
                Value : businessArea,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_hlException',
                Value : INCLUDE_hlException,
            },
            {
                $Type : 'UI.DataField',
                Label : 'hlException',
                Value : hlException,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_diagnosis',
                Value : INCLUDE_diagnosis,
            },
            {
                $Type : 'UI.DataField',
                Label : 'diagnosis',
                Value : diagnosis,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_systemResponse',
                Value : INCLUDE_systemResponse,
            },
            {
                $Type : 'UI.DataField',
                Label : 'systemResponse',
                Value : systemResponse,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_procedure',
                Value : INCLUDE_procedure,
            },
            {
                $Type : 'UI.DataField',
                Label : 'procedure',
                Value : procedure,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_referredMasterData',
                Value : INCLUDE_referredMasterData,
            },
            {
                $Type : 'UI.DataField',
                Label : 'referredMasterData',
                Value : referredMasterData,
            },
            {
                $Type : 'UI.DataField',
                Label : 'INCLUDE_referredConfiguration',
                Value : INCLUDE_referredConfiguration,
            },
            {
                $Type : 'UI.DataField',
                Label : 'referredConfiguration',
                Value : referredConfiguration,
            },
        ],
    },
    UI.Facets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Occurrences over Time',
            ID : 'OccurrencesoverTime',
            Target : 'timeAggregates/@UI.Chart#chartSection',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Daily Occurrences',
            ID : 'DailyOccurrences',
            Target : 'timeAggregates/@UI.LineItem#DailyOccurrences',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Prompt Settings',
            ID : 'PromptSettings',
            Target : '@UI.FieldGroup#PromptSettings',
        },
    ],
    UI.LineItem : [
        {
            $Type : 'UI.DataField',
            Label : 'ID',
            Value : ID,
        },
        {
            $Type : 'UI.DataField',
            Label : 'MSGNO',
            Value : MSGNO,
        },
        {
            $Type : 'UI.DataField',
            Label : 'messageClass',
            Value : messageClass,
        },
        {
            $Type : 'UI.DataField',
            Label : 'messageShortText',
            Value : messageShortText,
        },
        {
            $Type : 'UI.DataFieldForAnnotation',
            Target : 'timeAggregates/@UI.Chart#messageCount1',
            Label : 'messageCount',
        },
    ],
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'General Information',
            ID : 'GeneralInformation',
            Target : '@UI.FieldGroup#GeneralInformation',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'History',
            ID : 'History',
            Target : '@UI.FieldGroup#History',
        },
    ],
    UI.FieldGroup #GeneralInformation : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : MSGNO,
                Label : 'Message Number',
            },
            {
                $Type : 'UI.DataField',
                Value : messageClass,
                Label : 'Message Class',
            },
            {
                $Type : 'UI.DataField',
                Value : messageShortText,
                Label : 'Message Text',
            },
        ],
    },
    UI.FieldGroup #History : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : createdAt,
            },
            {
                $Type : 'UI.DataField',
                Value : createdBy,
            },
            {
                $Type : 'UI.DataField',
                Value : modifiedAt,
            },
            {
                $Type : 'UI.DataField',
                Value : modifiedBy,
            },
        ],
    },
    UI.FieldGroup #PromptSettings : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : INCLUDE_comments,
                Label : 'Include Domain Expert Comment?',
            },
            {
                $Type : 'UI.DataField',
                Value : comments,
                Label : 'Expert Instructions',
            },
        ],
    },
    UI.SelectionPresentationVariant #table : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.LineItem',
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
            ],
        },
    },
    UI.Identification : [
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'ReportService.assessRelated',
            Label : 'Regenerate related Assessments',
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'ReportService.generateReasoningSteps',
            Label : 'Generate Reasoning Steps',
        },
    ],
    UI.SelectionFields : [
        messageClass,
        MSGNO,
        priorityDeterminationStrategy,
        hasHappened,
    ],
    UI.SelectionPresentationVariant #table1 : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.LineItem',
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
                {
                    $Type : 'UI.SelectOptionType',
                    PropertyName : hasHappened,
                    Ranges : [
                        {
                            Sign : #I,
                            Option : #EQ,
                            Low : true,
                        },
                    ],
                },
            ],
        },
    },
);