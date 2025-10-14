import dotenv
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import Message, SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService


env = dotenv.load_dotenv()


ORCHESTRATION_SERVICE = OrchestrationService(deployment_id="d9e0c3ab7413a199")

class ChatBot:
    def __init__(self, orchestration_service=ORCHESTRATION_SERVICE, system_message=""):

        self.service = orchestration_service
        self.config = OrchestrationConfig(
            template=Template(
                messages=[
                    SystemMessage(system_message),
                    UserMessage("{{?user_query}}"),
                ],
            ),
            llm=LLM(name="gpt-4.1"),
        )

    def chat(self, user_input):

        response = self.service.run(
            config=self.config,
            template_values=[TemplateValue(name="user_query", value=user_input)],
        )
        message = response.orchestration_result.choices[0].message

        return message.content

