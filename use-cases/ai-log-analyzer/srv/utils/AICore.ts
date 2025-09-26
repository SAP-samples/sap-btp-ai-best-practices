import { SafeResponse } from "./types";

const cds = require('@sap/cds');

const LOG = cds.log('code', { label: 'code' })

const AI_CORE_DESTINATION = "GENERATIVE_AI_HUB";

export async function safeSendToAICore<T>(payload: any) {
    const aiCoreService = await cds.connect.to(AI_CORE_DESTINATION);
    LOG.debug(aiCoreService)

    const headers = {
        "Content-Type": "application/json",
        "AI-Resource-Group": aiCoreService.options.RESOURCE_GROUP_ID,
    };

    let safeResponse: SafeResponse<T>;

    try {
        const response = await aiCoreService.send({
            // @ts-ignore
            query: `POST /inference/deployments/${aiCoreService.options.DEPLOYMENT_ID}/chat/completions?api-version=${aiCoreService.options.API_VERSION}`,
            data: payload,
            headers: headers,
        });
        
        const parsedData = JSON.parse(response["choices"][0]?.message?.content) as T;

        safeResponse = {
            success: true,
            response: response,
            parsedData: parsedData
        }

    } catch (ex) {
        LOG.warn(`Exception processing ai core call`, ex)
        safeResponse = {success: false}
    }
    return safeResponse;
}