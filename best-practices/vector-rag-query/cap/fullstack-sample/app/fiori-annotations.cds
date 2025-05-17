using {RagService} from '../srv/rag-service';

////////////////////////////////////////////////////////////////////////////
//
//	Object Page
//
annotate RagService.ScienceData with @(UI: {
    HeaderInfo         : {
        TypeName      : 'ScienceData',
        TypeNamePlural: 'ScienceData',
        Description   : {Value: ID}
    },
    HeaderFacets       : [{
        $Type : 'UI.ReferenceFacet',
        Label : 'Classification',
        Target: '@UI.FieldGroup#Header'
    }, ],
    Facets             : [{
        $Type : 'UI.ReferenceFacet',
        Label : 'Topic',
        Target: '@UI.FieldGroup#Details'
    }, ],
    FieldGroup #Details: {Data: [{Value: Topic }, ]},
    FieldGroup #Header : {Data: [
        {
            Label: 'Category',
            Value: Category
        },
        {
            Label: 'Difficulty Level',
            Value: DifficultyLevel
        }
    ]},
});

////////////////////////////////////////////////////////////////////////////
//
//	List Report Page
//
annotate RagService.ScienceData with @(UI: {
    SelectionFields: [
        Category,
        DifficultyLevel
    ],
    LineItem       : [
        {
            Value                : ID,
            ![@HTML5.CssDefaults]: {
                $Type: 'HTML5.CssDefaultsType',
                width: '15rem'
            },
        },
        {
            Value                : Category,
            ![@HTML5.CssDefaults]: {
                $Type: 'HTML5.CssDefaultsType',
                width: '8rem'
            },
        },
        {
            Value                : DifficultyLevel,
            ![@HTML5.CssDefaults]: {
                $Type: 'HTML5.CssDefaultsType',
                width: '8rem'
            },
        },
        {
            Value                : Topic,
            ![@HTML5.CssDefaults]: {
                $Type: 'HTML5.CssDefaultsType',
                width: '100%'
            },
        },
    ],
});

annotate RagService.ScienceData with
@Capabilities.SearchRestrictions: {Searchable: false};

annotate RagService.ScienceData with {
    Category @Common.ValueList               : {
        CollectionPath: 'categoryList',
        Label         : 'Category',
        Parameters    : [{
            $Type            : 'Common.ValueListParameterInOut',
            LocalDataProperty: Category,
            ValueListProperty: 'Category'
        }],
    }
             @Common.ValueListWithFixedValues: true;

    DifficultyLevel
             @Common.ValueList               : {
        CollectionPath: 'difficultyLevelList',
        Label         : 'Difficulty Level',
        Parameters    : [{
            $Type            : 'Common.ValueListParameterInOut',
            LocalDataProperty: DifficultyLevel,
            ValueListProperty: 'DifficultyLevel'
        }],
    }
             @Common.ValueListWithFixedValues: true;
};
