from typing import List

import json
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import Message, SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from .chatbot import ChatBot
from .mail_classes import Email


def summarize_mail(mail_content: str):
    """
    Create a summary of an email.

    Args:
        mail_content: String containing the email information as HTML/text

    Returns:
        Only email summary
    """
    sys_prompt = """You are a helpful assistant that summarizes the content and meaning of emails
    down to short and concise description that captures the most vital and essential action items, problems
    and topics of the email.
    You must return the summary and nothing more.
    You will receive the mail as input """

    try:
        summarizer = ChatBot(system_message=sys_prompt)
        return summarizer.chat(mail_content)
    except Exception as e:
        # Log the error and return a fallback message
        print(f"Error in summarize_mail: {e}")
        return f"Error generating summary: {str(e)}"


def generate_response(mail_content: str, context: str = ""):
    """
    Write an email response.

    Args:
        mail_content: String containting the email infromation as json.
        context: String retrieved from S4 containing information on the invoice

    Returns:
        Only the response body of the email
    """
    sys_prompt = """You are a helpful assistant that crafts responses to emails given. Include all relevant information
    and make sure to write a helpful business formal response. Avoid slang, foul language and oversimplification.
    You must return the response and nothing more.
    You will receive the mail as input 
    Use the following content if provided: {}""".format(
        context if context else "No additional context provided"
    )

    try:
        response_generator = ChatBot(system_message=sys_prompt)
        return response_generator.chat(mail_content)
    except Exception as e:
        # Log the error and return a fallback message
        print(f"Error in generate_response: {e}")
        return f"Error generating response: {str(e)}"


def get_thread(id: str, mails: List[Email]):
    return [mails for mail in mails if mail.thread_id == id]


def get_folder(folder: str, mails: List[Email]):
    return [mails for mail in mails if mail.folder == folder]


def read_json(path: str) -> List:
    inbox = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    for email in emails:
        email_obj = Email.from_dict(email)
        inbox.append(email_obj)
    return email_obj


def write_to_json(inbox: List[Email], path: str):
    email_dicts = [email.to_dict for email in inbox]
    with open(path, "w") as f:
        json.dump(email_dicts, f, indent=2)


def append_mail(email: Email, path: str):
    inbox = read_json(path)
    inbox.append(email)
    write_to_json(inbox, path)
