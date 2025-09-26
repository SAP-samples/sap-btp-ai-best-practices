import cds from '@sap/cds';
import { JSONSchema, JSONSchemaProperties, JSONSchemaType, Tool, Tools } from '#cds-models/sap/btp/ai/tools';
import Ajv from 'ajv';
import { getAffectedProducts } from './tools/get_affected_products';
import { getAffectedReplenishmentRuns } from './tools/get_affected_replenishment_runs';
import { getAffectedStores } from './tools/get_affected_stores';
import { getAffectedWarehouses } from './tools/get_affected_warehouses';

const ajv = new Ajv({ removeAdditional:true });

const LOG = cds.log('code', { label: 'code' })

const toolFunctionMap: Record<string,Function> = {
    'get_affected_products': getAffectedProducts,
    'get_affected_replenishment_runs': getAffectedReplenishmentRuns,
    'get_affected_stores': getAffectedStores,
    'get_affected_warehouses': getAffectedWarehouses
}

export class ToolService extends cds.ApplicationService {
    async init(): Promise<void> {
        await super.init();
        this.on("call", (req) => {
            const { name }        = req.params[0] as { name: string };
            const { parameters }  = req.data       as { parameters: string };
            return this.call(name, parameters);
        });
    }

    public call = async (name: string, inputParametersRaw: string) => {
        LOG.info(`Calling tool '${name}' with parameters :: ${inputParametersRaw}`);
        const result: Tools = await cds.db.run(SELECT.from(Tools).where({ name: name}));
        if(result.length != 1) {
            throw new Error(`Tool with name ${name} not found.`);
        }
        const tool: Tool = result[0];
        LOG.info("Tool found :: ", tool);

        
        try {
                const parameters = this._parseAndValidateToolInput(tool, inputParametersRaw);
                const toolFunction = toolFunctionMap[name];
                const tool_output = await toolFunction(parameters)
                const output = this._validateToolOutput(tool, tool_output);

                LOG.info(`tool '${name}' invoked successfully :: `, output);
                return output;

        } catch (e) {
            LOG.warn(`Tool '${name}' invocation failed.`, e);
        }
    }

    private _parseAndValidateToolInput(tool: Tool, input:string) {
        try{
            const schema: object = {type : tool.inputSchema_type, properties: JSON.parse(tool.inputSchema_properties as string), required: tool.inputSchema_required}; 
            LOG.info(`Parsing and validating tool input against schema :: `, input, schema)
            const parameters = this._validate(this._parseObject(input), schema);
            return parameters;
        } catch (e) {
            LOG.warn(`Tool invocation failed because input could not be parsed or validated against input schema.`, e);
            throw e;
        }
    }

    private _validateToolOutput(tool: Tool, output:object) {
        try{
            const schema: object = {type : tool.outputSchema_type, properties: JSON.parse(tool.outputSchema_properties as string), required: tool.outputSchema_required}; 
            LOG.debug(`Parsing and validating tool output against schema :: `, output, schema)
            const result = this._validate(output, schema);
            return result;
        } catch (e) {
            LOG.warn(`Tool invocation failed because output does not match output schema.`, e);
            throw e;
        }
    }

    private _parseObject(raw: string) {
        LOG.debug(`Parsing: ${raw}`);
        try {
            const result = JSON.parse(raw) as object;
            LOG.debug(`Parsed`);
            return result;
          } catch (err) {
            LOG.error(`JSON parse failed`, err);
            throw new Error(`Invalid JSON: ${(err as Error).message}`);
          }
    }

    private _validate(object: object, schema: JSONSchema): /* validated */ object {
        LOG.debug(`Validating object against schema`, object, schema)
        if (!ajv.validate(schema as object, object)) {
            const msg = ajv.errorsText();
            LOG.error(`Schema validation failed: ${msg}`);
            throw new Error(msg);
          }
          LOG.info(`Validated ✔︎`);
          return object;
    }
}

