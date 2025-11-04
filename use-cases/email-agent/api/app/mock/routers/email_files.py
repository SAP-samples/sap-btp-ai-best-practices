from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
import mimetypes
from starlette.background import BackgroundTask
import time
import uuid

from ...security import get_api_key
from ...utils.attachment_extractor import (
    extract_email_metadata,
    AttachmentExtractor,
)


router = APIRouter(
    prefix="/email-files",
    tags=["email-files-mock"],
    dependencies=[Depends(get_api_key)],
)


def _emails_dir() -> Path:
    # This file is at app/mock/routers/email_files.py
    # We need to reach app/data/emails
    current = Path(__file__).resolve()
    emails_dir = current.parent.parent.parent / "data" / "emails"
    return emails_dir


def _attachments_dir() -> Path:
    # app/mock/routers -> app
    current = Path(__file__).resolve()
    return current.parent.parent.parent / "data" / "attachments"


@router.get("/", response_model=List[Dict[str, str]])
def list_msg_files() -> List[Dict[str, str]]:
    """List .msg files under app/data/emails (mock/demo convenience)."""
    emails_dir = _emails_dir()
    if not emails_dir.exists() or not emails_dir.is_dir():
        raise HTTPException(status_code=500, detail="Emails directory not found")

    results: List[Dict[str, str]] = []
    for entry in sorted(emails_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and entry.suffix.lower() == ".msg":
            try:
                results.append(
                    {
                        "name": entry.name,
                        "path": str(entry.resolve()),
                        "sizeBytes": str(entry.stat().st_size),
                    }
                )
            except Exception:
                continue
    return results


@router.get("/metadata")
def get_email_metadata(
    path: str = Query(..., description="Absolute path to email file")
) -> Dict[str, Any]:
    """Return basic metadata (subject, from, date, etc.) for a given email file path."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Email file not found")
    try:
        md = extract_email_metadata(str(p))
        # Normalize a few common keys for UI convenience
        return {
            "path": str(p.resolve()),
            "name": p.name,
            "from": md.get("from"),
            "subject": md.get("subject"),
            "date": (str(md.get("date")) if md.get("date") is not None else None),
            "attachments_count": md.get("attachments_count", 0),
            "attachment_filenames": md.get("attachment_filenames", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {exc}")


@router.get("/content")
def get_email_content(
    path: str = Query(..., description="Absolute path to email file")
) -> Dict[str, Any]:
    """Return original email content (html/text) and attachment filenames for preview."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Email file not found")
    try:
        suffix = p.suffix.lower()
        extractor = AttachmentExtractor()
        subject = None
        sender = None
        date = None
        body_html: Optional[str] = None
        body_text: Optional[str] = None

        def _ensure_str(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if isinstance(value, bytes):
                # Try common encodings encountered in Outlook messages; fall back permissively
                for enc in ("utf-8", "cp1252", "latin-1"):
                    try:
                        return value.decode(enc, errors="ignore")
                    except Exception:
                        continue
                try:
                    return value.decode(errors="ignore")
                except Exception:
                    return None
            try:
                return str(value)
            except Exception:
                return None

        if suffix == ".eml":
            msg = extractor.read_eml_file(str(p))
            subject = msg.get("Subject")
            sender = msg.get("From")
            date = msg.get("Date")
            # Extract html or text parts
            try:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True) or b""
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="ignore")
                        break
                if body_html is None:
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True) or b""
                            charset = part.get_content_charset() or "utf-8"
                            body_text = payload.decode(charset, errors="ignore")
                            break
            except Exception:
                pass

            filenames: List[str] = []
            try:
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        fname = part.get_filename()
                        if fname:
                            filenames.append(fname)
            except Exception:
                pass

        elif suffix == ".msg":
            msg = extractor.read_msg_file(str(p))
            # Coerce potentially-bytes fields to text to avoid serialization errors
            subject = _ensure_str(getattr(msg, "subject", None))
            sender = _ensure_str(getattr(msg, "sender", None))
            date = getattr(msg, "date", None)
            body_html = _ensure_str(getattr(msg, "htmlBody", None))
            body_text = _ensure_str(getattr(msg, "body", None)) or _ensure_str(
                getattr(msg, "stringBody", None)
            )

            filenames: List[str] = []
            try:
                if hasattr(msg, "attachments") and msg.attachments:
                    for att in msg.attachments:
                        raw_name = getattr(att, "longFilename", None) or getattr(
                            att, "shortFilename", None
                        )
                        fname = _ensure_str(raw_name)
                        if fname:
                            filenames.append(fname)
            except Exception:
                pass
        else:
            # Fallback for text-like files
            try:
                body_text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                body_text = None
            filenames = []

        return {
            "path": str(p.resolve()),
            "name": p.name,
            "subject": subject,
            "from": sender,
            "date": (str(date) if date is not None else None),
            "bodyHtml": body_html,
            "bodyText": body_text,
            "attachment_filenames": filenames,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to read email content: {exc}"
        )


@router.get("/attachment")
def get_email_attachment(
    path: str = Query(..., description="Absolute path to email file"),
    filename: Optional[str] = Query(None, description="Attachment filename to return"),
):
    """Extract and return a specific attachment from the given email file.

    If filename is not provided and there's only one attachment, returns that one.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Email file not found")
    try:
        out_dir = _attachments_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Only extract the requested attachment (avoid saving all)
        suffix = p.suffix.lower()
        extractor = AttachmentExtractor()
        saved_path: Optional[Path] = None
        chosen_name: Optional[str] = None

        # Helpers for safe filenames and unique paths
        def _sanitize_filename(name: str) -> str:
            invalid = '<>:"/\\|?*'
            for ch in invalid:
                name = name.replace(ch, "_")
            return name.strip(" .") or "attachment.bin"

        def _unique_path(directory: Path, name: str) -> Path:
            base = Path(_sanitize_filename(name))
            candidate = directory / base
            if not candidate.exists():
                return candidate
            stem = base.stem
            suffix = base.suffix
            # Add short uuid tail to avoid many retries
            return directory / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"

        if suffix == ".msg":
            msg = extractor.read_msg_file(str(p))
            # Build a list of available attachment names
            available = []
            if hasattr(msg, "attachments") and msg.attachments:
                for att in msg.attachments:
                    fname = getattr(att, "longFilename", None) or getattr(
                        att, "shortFilename", None
                    )
                    if fname:
                        available.append((fname, att))
            if not available:
                raise HTTPException(status_code=404, detail="No attachments found")

            chosen_att = None
            if filename:
                for fname, att in available:
                    if fname == filename:
                        chosen_att = att
                        chosen_name = fname
                        break
                if not chosen_att:
                    raise HTTPException(status_code=404, detail="Attachment not found")
            else:
                if len(available) == 1:
                    chosen_name, chosen_att = available[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Multiple attachments; specify filename query parameter",
                    )

            # Save only the chosen attachment
            safe_name = _sanitize_filename(chosen_name or "attachment.bin")
            target_path = _unique_path(out_dir, safe_name)
            chosen_att.save(
                customPath=str(target_path.parent), customFilename=target_path.name
            )
            saved_path = target_path

        elif suffix == ".eml":
            msg = extractor.read_eml_file(str(p))
            # Collect attachment parts
            parts = []
            try:
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        parts.append(part)
            except Exception:
                pass
            if not parts:
                raise HTTPException(status_code=404, detail="No attachments found")

            chosen_part = None
            if filename:
                for part in parts:
                    if part.get_filename() == filename:
                        chosen_part = part
                        chosen_name = filename
                        break
                if not chosen_part:
                    raise HTTPException(status_code=404, detail="Attachment not found")
            else:
                if len(parts) == 1:
                    chosen_part = parts[0]
                    chosen_name = chosen_part.get_filename() or "attachment.bin"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Multiple attachments; specify filename query parameter",
                    )

            payload = chosen_part.get_payload(decode=True) or b""
            safe_name = _sanitize_filename(chosen_name or "attachment.bin")
            target_path = _unique_path(out_dir, safe_name)
            saved_path = target_path
            with open(saved_path, "wb") as f:
                f.write(payload)

        else:
            raise HTTPException(status_code=400, detail="Unsupported email file type")

        file_path = str(saved_path) if saved_path else None
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Attachment file missing")

        media_type, _ = mimetypes.guess_type(file_path)
        media_type = media_type or "application/octet-stream"
        disp_name = Path(file_path).name
        headers = {"Content-Disposition": f'inline; filename="{disp_name}"'}

        # Ensure cleanup runs after the response has been fully sent; retry if file is busy
        def _cleanup_attachment(p: str):
            for _ in range(6):
                try:
                    Path(p).unlink(missing_ok=True)
                    break
                except Exception:
                    time.sleep(0.5)

        cleanup = BackgroundTask(_cleanup_attachment, file_path)
        return FileResponse(
            file_path, media_type=media_type, headers=headers, background=cleanup
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to return attachment: {exc}"
        )
