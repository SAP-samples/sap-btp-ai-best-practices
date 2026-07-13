const cds = require('@sap/cds');

const LOG = cds.log('code', { label: 'code' })

const AI_CORE_DESTINATION = "GENERATIVE_AI_HUB";

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

export async function retry<T>(times: number, func: () => Promise<SafeResponse<T>>): Promise<SafeResponse<T>> {
  let lastError: any;
  while (times > 0) {
    try {
      const result = await func();
      if (result.success) {
        return result;
      } else {
        lastError = new Error("SafeResponse indicates failure, retrying.");
      }
    } catch (error) {
      LOG.debug(`times :: ${times}`, error)
    }
    times--;
    // delay between retries
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  LOG.warn(`Last retry failed with error.`, lastError)
  return { success: false };
}

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