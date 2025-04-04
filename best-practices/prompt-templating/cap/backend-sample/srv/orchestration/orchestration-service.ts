import { OrchestrationClient } from '@sap-ai-sdk/orchestration';

export default class OrchestrationService {
  async askCapitalOfCountry(req: any) {
    const country = req.data.country;
    if (!country) {
      req.reject(400, 'Country parameter is required');
    }
    const orchestrationClient = new OrchestrationClient({
      llm: {
        model_name: 'gpt-4o',
        model_params: {
          max_tokens: 1000,
          temperature: 0.6,
          n: 1
        }
      },
      templating: {
        template: [
          { role: 'user', content: `What is the capital of {{?country}}?` }
        ]
      }
    });

    const result = await orchestrationClient.chatCompletion({
      inputParams: { country: country }
    });
    return result.getContent();
  }
}
