using ReportService as service from '../../srv/report-service';

annotate service.MessageTimeAggregates with @(
    UI.FieldGroup #GeneratedGroup : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Label : 'ID',
                Value : ID,
            },
            {
                $Type : 'UI.DataField',
                Label : 'timeFrom',
                Value : timeFrom,
            },
            {
                $Type : 'UI.DataField',
                Label : 'timeTo',
                Value : timeTo,
            },
            {
                $Type : 'UI.DataField',
                Label : 'situation_ID',
                Value : situation_ID,
            },
            {
                $Type : 'UI.DataField',
                Label : 'messageCount',
                Value : messageCount,
            },
        ],
    },
    UI.Facets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Assessment',
            ID : 'Assessment',
            Target : '@UI.FieldGroup#Assessment',
        },
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Log Messages',
            ID : 'LogMessages',
            Target : 'messages/@UI.LineItem#LogMessages',
        },
    ],
    UI.LineItem : [
        {
            $Type : 'UI.DataField',
            Value : date,
            Label : 'Report Date',
        },
        {
            $Type : 'UI.DataField',
            Value : situation.messageClass,
            Label : 'Message Class',
        },
        {
            $Type : 'UI.DataField',
            Value : situation.MSGNO,
            Label : 'Message Number',
            
        },
        {
            $Type : 'UI.DataField',
            Label : 'Message Count',
            Value : messageCount,
        },
        {
            $Type : 'UI.DataField',
            Value : latestAssessment.priority,
            Label : 'Assessed Priority',
            Criticality : latestAssessment.priorityCriticality,
        },
        {
            $Type : 'UI.DataField',
            Value : latestAssessment.summary,
            Label : 'AI Summary',
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'ReportService.assess',
            Label : 'Assess',
            Inline : true,
        },
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'ReportService.bulkAssess',
            Label : 'Bulk Assessment',
        },
    ],
    UI.FieldGroup #Assessment : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : latestAssessment.nextSteps,
                Label : 'Next Steps',
            },
            {
                $Type : 'UI.DataField',
                Value : latestAssessment.configurationChanges,
                Label : 'Configuration Changes',
            },
            {
                $Type : 'UI.DataField',
                Value : latestAssessment.rootCause,
                Label : 'Root Cause',
            },
        ],
    },
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Situation Details',
            ID : 'SituationDetails',
            Target : '@UI.FieldGroup#SituationDetails',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'date',
            Target : '@UI.DataPoint#date',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'priority',
            Target : 'latestAssessment/@UI.DataPoint#priority',
        },
        {
            $Type : 'UI.ReferenceFacet',
            ID : 'summary',
            Target : 'latestAssessment/@UI.DataPoint#summary',
        },
    ],
    UI.FieldGroup #SituationDetails : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : situation.MSGNO,
                Label : 'Message Number',
            },
            {
                $Type : 'UI.DataField',
                Value : situation.messageClass,
                Label : 'Message Class',
            },
        ],
    },
    UI.LineItem #DailyOccurrences : [
        {
            $Type : 'UI.DataField',
            Value : ID,
            Label : 'ID',
        },
        {
            $Type : 'UI.DataField',
            Value : date,
            Label : 'Date',
        },
        {
            $Type : 'UI.DataField',
            Value : latestAssessment.priority,
            Label : 'Assessed Priority',
            Criticality : latestAssessment.priorityCriticality
        },
        {
            $Type : 'UI.DataField',
            Value : messageCount,
            Label : 'Message Count',
        },
    ],
    UI.FieldGroup #Summary : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : latestAssessment.priority,
                Label : 'priority',
                Criticality : latestAssessment.priorityCriticality
            },
            {
                $Type : 'UI.DataField',
                Value : latestAssessment.summary,
                Label : 'summary',
            },
        ],
    },
    UI.DataPoint #date : {
        $Type : 'UI.DataPointType',
        Value : date,
        Title : 'Date',
    },
    Analytics.AggregatedProperty #messageCount_sum : {
        $Type : 'Analytics.AggregatedPropertyType',
        Name : 'messageCount_sum',
        AggregatableProperty : messageCount,
        AggregationMethod : 'sum',
        ![@Common.Label] : 'Message Count (Sum)',
    },
    UI.Chart #chartSection : {
        $Type : 'UI.ChartDefinitionType',
        ChartType : #Column,
        Dimensions : [
            date,
        ],
        DynamicMeasures : [
            '@Analytics.AggregatedProperty#messageCount_sum',
        ],
    },
    UI.DataPoint #messageCount : {
        Value : messageCount,
    },
    UI.Chart #messageCount : {
        ChartType : #Line,
        Measures : [
            messageCount,
        ],
        MeasureAttributes : [
            {
                DataPoint : '@UI.DataPoint#messageCount',
                Role : #Axis1,
                Measure : messageCount,
            },
        ],
        Dimensions : [
            date,
        ],
    },
    UI.DataPoint #messageCount1 : {
        Value : messageCount,
    },
    UI.Chart #messageCount1 : {
        ChartType : #Line,
        Measures : [
            messageCount,
        ],
        MeasureAttributes : [
            {
                DataPoint : '@UI.DataPoint#messageCount1',
                Role : #Axis1,
                Measure : messageCount,
            },
        ],
        Dimensions : [
            date,
        ],
    },
    UI.Identification : [
        {
            $Type : 'UI.DataFieldForAction',
            Action : 'ReportService.assess',
            Label : 'Regenerate Assessment',
        },
    ],
    UI.SelectionFields : [
        date,
        latestAssessment.priority,
        situation.messageClass,
        situation.MSGNO,
    ],
    UI.SelectionPresentationVariant #table : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.LineItem',
            ],
            SortOrder : [
                {
                    $Type : 'Common.SortOrderType',
                    Property : date,
                    Descending : true,
                },
                {
                    $Type : 'Common.SortOrderType',
                    Property : messageCount,
                    Descending : true,
                },
            ],
            GroupBy : [
                date,
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
            ],
        },
    },
    UI.Chart #alpChart : {
        $Type : 'UI.ChartDefinitionType',
        ChartType : #ColumnStacked,
        Dimensions : [
            date,
            latestAssessment_priorityCriticality,
        ],
        DynamicMeasures : [
            '@Analytics.AggregatedProperty#messageCount_sum',
        ],
        Title : 'Total Messages by Day',
        DimensionAttributes : [
            {
                Dimension : latestAssessment_priorityCriticality,
                Role : #Series,
            },
        ],
        MeasureAttributes : [
            {
                $Type          : 'UI.ChartMeasureAttributeType',
                Measure : '@Analytics.AggregatedProperty#messageCount_sum',
                // Role           : #Axis1,
                DataPoint      : '@UI.DataPoint#MsgCnt'
            }
        ]
    },
    UI.DataPoint #MsgCnt : {
        $Type     : 'UI.DataPointType',
        Title     : 'Message Count',
        Value     : '@Analytics.AggregatedProperty#messageCount_sum',
        Criticality : latestAssessment_priorityCriticality
    },
    UI.SelectionPresentationVariant #alpChart : {
        $Type : 'UI.SelectionPresentationVariantType',
        PresentationVariant : {
            $Type : 'UI.PresentationVariantType',
            Visualizations : [
                '@UI.Chart#alpChart',
            ],
            SortOrder : [
                {
                    $Type : 'Common.SortOrderType',
                    Descending : false,
                    Property : date,
                },
            ],
        },
        SelectionVariant : {
            $Type : 'UI.SelectionVariantType',
            SelectOptions : [
            ],
        },
    }
);

annotate service.MessageTimeAggregates with {
    situation @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Situations',
        Parameters : [
            {
                $Type : 'Common.ValueListParameterInOut',
                LocalDataProperty : situation_ID,
                ValueListProperty : 'ID',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'MSGNO',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'messageClass',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'messageShortText',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'messageType',
            },
        ],
    }
};

annotate service.MessageTimeAggregates with {
    latestAssessment @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Assessments',
        Parameters : [
            {
                $Type : 'Common.ValueListParameterInOut',
                LocalDataProperty : latestAssessment_ID,
                ValueListProperty : 'ID',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'prompt',
            },
            {
                $Type : 'Common.ValueListParameterDisplayOnly',
                ValueListProperty : 'timeAggregate_ID',
            },
        ],
    }
};

annotate service.Assessments with @(
    UI.LineItem #PreviosAssessments : [
        {
            $Type : 'UI.DataField',
            Value : ID,
            Label : 'ID',
        },
        {
            $Type : 'UI.DataField',
            Value : priority,
            Label : 'Priority',
        },
        {
            $Type : 'UI.DataField',
            Value : createdAt,
        },
        {
            $Type : 'UI.DataField',
            Value : createdBy,
        },
    ],
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'General',
            ID : 'General',
            Target : '@UI.FieldGroup#General',
        },
    ],
    UI.FieldGroup #General : {
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
    UI.Facets : [
        
    ],
    UI.DataPoint #priority : {
        $Type : 'UI.DataPointType',
        Value : priority,
        Title : 'Assessed Priority',
        Criticality : priorityCriticality,
    },
    UI.DataPoint #summary : {
        $Type : 'UI.DataPointType',
        Value : summary,
        Title : 'Summary',
    },
);

annotate service.Messages with @(
    UI.LineItem #LogMessages : [
        {
            $Type : 'UI.DataField',
            Value : REPL_RUN_ID,
            Label : 'REPL_RUN_ID',
        },
        {
            $Type : 'UI.DataField',
            Value : productDescription,
            Label : 'productDescription',
        },
        {
            $Type : 'UI.DataField',
            Value : formattedMessage,
            Label : 'Message',
        },
        {
            $Type : 'UI.DataField',
            Value : locationFrom.name,
            Label : 'Location From',
        },
        {
            $Type : 'UI.DataField',
            Value : location.name,
            Label : 'Location',
        },
    ],
    UI.HeaderFacets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Message Information',
            ID : 'MessageInformation',
            Target : '@UI.FieldGroup#MessageInformation',
        },
    ],
    UI.FieldGroup #MessageInformation : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : product.description,
                Label : 'Product',
            },
            {
                $Type : 'UI.DataField',
                Value : locationFrom.name,
                Label : 'Location From',
            },
            {
                $Type : 'UI.DataField',
                Value : location.name,
                Label : 'Location',
            },
        ],
    },
    UI.Facets : [
        {
            $Type : 'UI.ReferenceFacet',
            Label : 'Message Context',
            ID : 'Context',
            Target : '@UI.FieldGroup#Context',
        },
    ],
    UI.Identification : [
    ],
    UI.FieldGroup #Context : {
        $Type : 'UI.FieldGroupType',
        Data : [
            {
                $Type : 'UI.DataField',
                Value : ALLO_PLAN_ID,
                Label : 'ALLO_PLAN_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : ALLO_WL_ID,
                Label : 'ALLO_WL_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : COLOR,
                Label : 'COLOR',
            },
            {
                $Type : 'UI.DataField',
                Value : CONTROLLER_ID,
                Label : 'CONTROLLER_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : COUNTER,
                Label : 'COUNTER',
            },
            {
                $Type : 'UI.DataField',
                Value : CREATIONTSTAMP,
                Label : 'Creation Timestamp',
            },
            {
                $Type : 'UI.DataField',
                Value : DELETION_INDICATOR,
                Label : 'Deletion Indicator',
            },
            {
                $Type : 'UI.DataField',
                Value : DELETIONTSTAMP,
                Label : 'Deletion Timestamp',
            },
            {
                $Type : 'UI.DataField',
                Value : DIAG_ID,
                Label : 'DIAG_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : DISTR_CHNL_ID,
                Label : 'DISTR_CHNL_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : FORECAST_ID,
                Label : 'FORECAST_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : ID,
                Label : 'ID',
            },
            {
                $Type : 'UI.DataField',
                Value : IS_ASSIGNED,
                Label : 'IS_ASSIGNED',
            },
            {
                $Type : 'UI.DataField',
                Value : INBOUND_CONTEXT,
                Label : 'INBOUND_CONTEXT',
            },
            {
                $Type : 'UI.DataField',
                Value : JOB_ID,
                Label : 'JOB_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : KPRM,
                Label : 'KPRM',
            },
            {
                $Type : 'UI.DataField',
                Value : LOCATION_HIERARCHY_ID,
                Label : 'LOCATION_HIERARCHY_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : LOCATION_ID,
                Label : 'LOCATION_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : LOCATIONFROM_ID,
                Label : 'LOCATIONFROM_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : MANDT,
                Label : 'MANDT',
            },
            {
                $Type : 'UI.DataField',
                Value : MARKET_UNIT_ID,
                Label : 'MARKET_UNIT_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : MOD,
                Label : 'MOD',
            },
            {
                $Type : 'UI.DataField',
                Value : MODEL_ID,
                Label : 'MODEL_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : MSGPSTAMP,
                Label : 'MSGPSTAMP',
            },
            {
                $Type : 'UI.DataField',
                Value : OFFER_ID,
                Label : 'OFFER_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : ORDER_CHNL_ID,
                Label : 'ORDER_CHNL_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : ORDERPLANITEM4AUUID,
                Label : 'ORDERPLANITEM4AUUID',
            },
            {
                $Type : 'UI.DataField',
                Value : ORDERPLANITEMUUID,
                Label : 'ORDERPLANITEMUUID',
            },
            {
                $Type : 'UI.DataField',
                Value : PROBCLASS,
                Label : 'PROBCLASS',
            },
            {
                $Type : 'UI.DataField',
                Value : PRODUCT_HIERARCHY_ID,
                Label : 'PRODUCT_HIERARCHY_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : PRODUCT_ID,
                Label : 'PRODUCT_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : PROG,
                Label : 'PROG',
            },
            {
                $Type : 'UI.DataField',
                Value : REFERENCE_ID,
                Label : 'REFERENCE_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : RELCONT,
                Label : 'RELCONT',
            },
            {
                $Type : 'UI.DataField',
                Value : REPL_RUN_ID,
                Label : 'REPL_RUN_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : RUNDATE,
                Label : 'RUNDATE',
            },
            {
                $Type : 'UI.DataField',
                Value : SALES_ORG_ID,
                Label : 'SALES_ORG_ID',
            },
            {
                $Type : 'UI.DataField',
                Value : STATUS,
                Label : 'STATUS',
            },
            {
                $Type : 'UI.DataField',
                Value : TCODE,
                Label : 'TCODE',
            },
            {
                $Type : 'UI.DataField',
                Value : USR,
                Label : 'USR',
            },
            {
                $Type : 'UI.DataField',
                Value : VENDOR_FUND_ID,
                Label : 'VENDOR_FUND_ID',
            },
        ],
    },
    UI.HeaderInfo : {
        TypeName : 'Message',
        TypeNamePlural : 'Messages',
        Title : {
            $Type : 'UI.DataField',
            Value : formattedMessage,
        },
        Description : {
            $Type : 'UI.DataField',
            Value : productDescription,
        },
    },
);

annotate service.Assessments with {
    prompt @UI.MultiLineText : true
};

annotate service.Assessments with {
    response @UI.MultiLineText : true
};

annotate service.MessageTimeAggregates with {
    date @Common.Label : 'Report Date'
};

annotate service.Assessments with {
    priority @Common.Label : 'Assessed Priority'
};



