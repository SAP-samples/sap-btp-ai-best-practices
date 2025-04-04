import { OrchestrationClient } from '@sap-ai-sdk/orchestration';
import cds from '@sap/cds';

const LOG = cds.log('orchestration');

export default class OrchestrationService {
  async generateEmail(req: any) {
    const { prompt, anonymize } = req.data;
    const orchestrationClient = new OrchestrationClient({
      llm: {
        model_name: 'gpt-4-32k'
      },
      templating: {
        template: [
          {
            role: 'user',
            content: prompt
          }
        ]
      },
      masking: {
        masking_providers: [
          {
            type: 'sap_data_privacy_integration',
            method: anonymize ? 'anonymization' : 'pseudonymization',
            entities: [{ type: 'profile-email' }, { type: 'profile-person' }]
          }
        ]
      }
    });

    const result = await orchestrationClient.chatCompletion();
    LOG.info(result.rawResponse);
    return result.getContent();
  }
}
