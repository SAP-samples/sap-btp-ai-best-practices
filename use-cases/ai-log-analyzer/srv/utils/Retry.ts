
const cds = require('@sap/cds');
const LOG = cds.log('code', { label: 'code' })
import { SafeResponse } from './types';

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
