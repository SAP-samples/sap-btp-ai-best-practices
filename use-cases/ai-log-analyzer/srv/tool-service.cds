using { sap.btp.ai.tools as my } from '../db/tools-schema.cds';

@path : '/service/ToolService'
service ToolService
{
    @readonly
    entity Tools as projection on my.Tools {
        *
    } excluding {
        modifiedAt,
        createdAt,
        modifiedBy,
        createdBy
    } actions {
        action call(parameters: LargeString) returns String;
    }
}
