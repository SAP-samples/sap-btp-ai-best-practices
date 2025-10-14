from typing import List
import dotenv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import Message, SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from .chatbot import ChatBot
from .mail_classes import Email

STATUS_PROMPT = (
    "You are an AI assistant for a high-volume Accounts Payable (AP) department. Your task is to analyze an incoming email and classify its current status to help the team manage their workload"
    "Based on the content of the email, you must return **only one** of the following three statuses:"
    "*   `action-needed`"
    "*   `waiting-for-response`"
    "*   `resolved`"
    "**Classification Rules:**"
    "1.  **`action-needed`**: Assign this status if the email:"
    "*   Is a new inquiry from an internal or external source (e.g., asking for payment status, invoice information, or vendor updates)."
    "*   Contains a serious notification, such as a credit hold."
    "*   Is a response to a question from the AP team that provides the requested information, allowing the team to proceed."
    "*   Is a follow-up from the sender asking for an update on their original request."
    "2.  **`waiting-for-response`**: Assign this status **only** if the email's content explicitly shows that the AP team has already asked a question and is now waiting for a reply from the other party. This status is less common for a newly received email."
    "3.  **`resolved`**: Assign this status if the sender's email confirms that:"
    "*   Their issue has been fully addressed."
    "*   They have received the payment or information they were looking for."
    "*   The matter is closed (e.g., 'Thank you, this is all I needed,' or 'Problem solved')."
    "Your output must be **exactly** one of these three options and nothing else. Now, classify the following email:"
)
PRIORITY_PROMPT = """You are an AI assistant for a high-volume Accounts Payable (AP) department. Your task is to analyze an incoming email and classify its criticality to help the team manage their workload.
    Based on the content of the email, you must return exactly one of the following classification levels:

    critical
    high
    medium
    low

    Classification Rules:

    critical: Assign this level if the email:

    Contains words like: "urgent," "critical," "immediate," "emergency," "legal action," "lawsuit"
    References late payments, missed deadlines, or overdue amounts
    Contains words like: "past due," "overdue," "delinquent," "missed payment"
    Mentions account suspension, credit holds, or service termination
    References overdue payments with threats of consequences
    Involves regulatory compliance issues or audits


    high: Assign this level if the email:

    Contains words like: "final notice," "escalation," "reminder"
    Mentions multiple reminders or follow-ups
    References payment deadlines within 1-3 business days
    Involves key vendor relationships or large amounts
    Contains upcoming due dates with potential for becoming overdue


    medium: Assign this level if the email:

    Contains routine payment reminders or invoice inquiries
    References standard payment terms or upcoming due dates
    Involves vendor onboarding or account setup requests
    Contains general questions about payment status


    low: Assign this level if the email:

    Contains informational updates or announcements
    Involves routine correspondence with no immediate action needed
    References future payments or preliminary discussions
    Contains thank you messages or confirmations



    Your output must be exactly one of these four classification levels and nothing else. Now, classify the following email:"""
TAGS_PROMPT = """
    You are an AI email classifier for an Accounts Payable (AP) department. Your sole function is to analyze the content of an email and assign it to one of the seven categories below.

    ### Classification Categories:

    1.  **`Receiving of Invoice Notification`**
        *   Use this if the sender is notifying about a new invoice being sent, invoice delivery confirmation, or providing invoice details for processing. Look for keywords like "invoice attached", "invoice sent", "new invoice", "invoice notification", "invoice submitted", or "please process invoice".

    2.  **`Expedited Payment Request`**
        *   Use this if the sender is requesting urgent or accelerated payment processing due to special circumstances. Look for keywords like "urgent payment", "expedite", "rush payment", "immediate payment needed", "critical payment request", "emergency payment", or "time-sensitive payment".

    3.  **`Short Payments`**
        *   Use this if the email relates to payment discrepancies, partial payments, cash received without proper remittance details, or mismatched payment amounts. Look for keywords like "short payment", "partial payment", "underpaid", "payment discrepancy", "cash received with no remittance", "remittance detail that did not match", "amount difference", or "apply dollars to outstanding balances".

    4.  **`Payment Inquiries`**
        *   Use this if the sender is asking about payment status, invoice lookup requests, overdue payments, or account statements. Look for keywords like "payment status", "invoice lookup", "Ariba", "overdue", "past due", "check payment status", "payment timeline", "when will we be paid", "remittance", "outstanding balance", "payment received", or "significantly overdue".

    5.  **`Changes to Invoices`**
        *   Use this if the email relates to invoice modifications, corrections, credits, cancellations, or requests to revise already submitted invoices. Look for keywords like "invoice correction", "credit memo", "revised invoice", "invoice amendment", "invoice cancellation", "modify invoice", or "update invoice details".

    6.  **`Vendor data updates`**
        *   Use this if the email's main purpose is to provide updated vendor information including W-9 forms, tax ID changes, banking details, address changes, business name changes, or other official vendor documentation updates. Look for keywords like "W9", "W-9", "tax ID", "EIN", "address change", "business name change", "tax classification", "banking updates", "vendor information update", "business structure change", or "responsible party information".

    7.  **`Other/Unclear`**
        *   Use this for emails that do not clearly fall into the above six categories. This includes internal requests, documentation requests (SOPs), rerouting emails, resolved payment confirmations, thank you messages, spam, general inquiries, or messages that are too vague to classify. Look for keywords like "internal", "reroute", "SOP", "documentation", "resolved", "thank you", "payment received", "matter is closed", or "inquiry is now resolved".

    Your response must be **only** the exact category name and nothing else.

    Now, classify the following email:
    """


def classify_status_file(path: str):

    mail_classifier = ChatBot(system_message=STATUS_PROMPT)

    try:
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    for email in emails:
        classification = mail_classifier.chat(email["body"]["html"])
        email["status"] = classification
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(emails, file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")


def classify_priority_file(path: str):
    mail_classifier = ChatBot(system_message=PRIORITY_PROMPT)

    try:
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    for email in emails:
        classification = mail_classifier.chat(email["body"]["html"])
        email["priority"] = classification.lower()
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(emails, file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")


def classify_tags_file(path: str):

    mail_classifier = ChatBot(system_message=TAGS_PROMPT)

    try:
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    for email in emails:
        classification = mail_classifier.chat(email["body"]["html"])
        email["tags"] = [classification]
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(emails, file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")


def classify_status_mail(mail: Email):

    mail_classifier = ChatBot(system_message=STATUS_PROMPT)

    return mail_classifier.chat(mail.body.html)


def classify_priority_mail(mail: Email):

    mail_classifier = ChatBot(system_message=PRIORITY_PROMPT)
    return mail_classifier.chat(mail.body.html)


def classify_tags_mail(mail: Email):
    mail_classifier = ChatBot(system_message=TAGS_PROMPT)

    return mail_classifier.chat(mail.body.html)


def classify_single_email(email_data):
    """Classify a single email with all three types (status, priority, tags) in parallel."""
    try:
        email_html = email_data["body"]["html"]

        # Create classifiers for each type
        status_classifier = ChatBot(system_message=STATUS_PROMPT)
        priority_classifier = ChatBot(system_message=PRIORITY_PROMPT)
        tags_classifier = ChatBot(system_message=TAGS_PROMPT)

        # Classify with all three types
        status = status_classifier.chat(email_html)
        priority = priority_classifier.chat(email_html).lower()
        tags = tags_classifier.chat(email_html)

        # Update the email data
        email_data["status"] = status
        email_data["priority"] = priority
        email_data["tags"] = [tags] if tags else []

        return email_data
    except Exception as e:
        print(f"Error classifying email {email_data.get('messageId', 'unknown')}: {e}")
        return email_data


def classify_emails_parallel(path: str, max_workers: int = 50):
    """Classify all emails in parallel for much faster processing."""
    try:
        # Load emails
        with open(path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Process emails in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all emails for processing
        future_to_email = {
            executor.submit(classify_single_email, email): i
            for i, email in enumerate(emails)
        }

        # Collect results as they complete
        for future in as_completed(future_to_email):
            email_index = future_to_email[future]
            try:
                classified_email = future.result()
                emails[email_index] = classified_email
            except Exception as e:
                print(f"Error processing email at index {email_index}: {e}")

    # Save the classified emails
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(emails, file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")
        raise
