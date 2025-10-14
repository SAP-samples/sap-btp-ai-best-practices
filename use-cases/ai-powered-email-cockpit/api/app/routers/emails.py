import json
import os
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from ..utils.classifier import (
    classify_status_file,
    classify_tags_file,
    classify_priority_file,
    classify_emails_parallel,
)
from ..utils.mail_classes import Email, Sender
from ..utils.mail_tools import generate_response, summarize_mail


router = APIRouter(prefix="/emails", tags=["emails"])


def _data_file_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # routers/ -> app/
    app_dir = os.path.dirname(current_dir)
    data_path = os.path.join(app_dir, "data", "business_emails.json")
    return data_path


def _load_emails() -> List[dict]:
    path = _data_file_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Emails data file not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read emails: {exc}")


def _get_email(message_id: str) -> dict:
    emails = _load_emails()
    for e in emails:
        if e.get("messageId") == message_id:
            return e
    raise HTTPException(status_code=404, detail="Email not found")


@router.get("/")
def list_emails(
    tag: Optional[str] = Query(default=None, description="Filter by tag")
) -> List[dict]:
    emails = _load_emails()
    if tag:
        tag_lower = tag.lower()
        emails = [
            e for e in emails if any(t.lower() == tag_lower for t in e.get("tags", []))
        ]
    # Sort by sentDate desc if present
    emails.sort(key=lambda e: e.get("sentDate", ""), reverse=True)
    return emails


@router.get("/tags")
def list_tags() -> List[str]:
    emails = _load_emails()
    tag_set = set()
    for e in emails:
        for t in e.get("tags", []) or []:
            if isinstance(t, str):
                tag_set.add(t)
    return sorted(tag_set, key=lambda s: s.lower())


@router.get("/{message_id}")
def get_email(message_id: str) -> dict:
    return _get_email(message_id=message_id)


@router.post("/classify")
def classify_emails() -> dict:
    """Trigger classification for tags and status over the data file.

    Uses utility functions in `utils.classifier` to update emails in-place.
    Returns a simple summary including the number of processed emails.
    """
    path = _data_file_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Emails data file not found")

    try:
        # Classify all emails in parallel - much faster!
        classify_emails_parallel(path)

        # Load updated emails to compute summary
        emails = _load_emails()
        count = len(emails)
        return {"success": True, "count": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")


@router.post("/reset")
def reset_email_classifications() -> dict:
    """Remove classifications (tags, status) from the data file for demo reset."""
    path = _data_file_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Emails data file not found")

    try:
        with open(path, "r", encoding="utf-8") as f:
            emails = json.load(f)

        for e in emails:
            e["tags"] = []
            e["status"] = ""
            e["priority"] = ""

        with open(path, "w", encoding="utf-8") as f:
            json.dump(emails, f, indent=2, ensure_ascii=False)

        return {"success": True, "count": len(emails)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


@router.post("/respond/{message_id}")
def generate_email_response(message_id: str):
    """Generate an AI response to an email."""
    logger = logging.getLogger(__name__)
    logger.info(f"Response request for message_id: {message_id}")

    try:
        # URL decode the message_id in case of special characters
        decoded_message_id = unquote(message_id)
        logger.info(f"Decoded message_id: {decoded_message_id}")

        email_data = _get_email(decoded_message_id)
        email = Email.from_dict(email_data)
        logger.info(f"Found email with subject: {email.subject}")
    except HTTPException as he:
        raise he
    except Exception as exc:
        logger.error(f"Email lookup failed for {message_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid email request")

    try:
        # Generate response using the email content
        response_content = generate_response(email.body.html)
        logger.info(
            f"Response generated successfully: {response_content[:100]}..."
            if response_content
            else "Empty response"
        )

        return {
            "success": True,
            "response": response_content,
            "original_subject": email.subject,
            "original_sender": email.sender.to_dict(),
        }
    except Exception as exc:
        logger.error(
            f"Response generation failed for {message_id}: {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Response generation failed: {exc}"
        )


@router.post("/summary/{message_id}")
def generate_email_summary(message_id: str):
    """Generate an AI summary of an email."""
    logger = logging.getLogger(__name__)
    logger.info(f"Summary request for message_id: {message_id}")

    try:
        # URL decode the message_id in case of special characters
        decoded_message_id = unquote(message_id)
        logger.info(f"Decoded message_id: {decoded_message_id}")

        email_data = _get_email(decoded_message_id)
        logger.info(f"Found email with subject: {email_data.get('subject', 'N/A')}")

        email = Email.from_dict(email_data)
        logger.info("Email object created successfully")

        summary = summarize_mail(email.body.html)
        logger.info(
            f"Summary generated successfully: {summary[:100]}..."
            if summary
            else "Empty summary"
        )

        return {
            "success": True,
            "summary": summary,
            "email_subject": email.subject,
            "email_sender": email.sender.to_dict(),
        }
    except HTTPException as he:
        # Re-raise HTTP exceptions (like 404)
        raise he
    except Exception as exc:
        logger.error(
            f"Summary generation failed for {message_id}: {exc}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc}")
