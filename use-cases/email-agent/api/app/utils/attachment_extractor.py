import os
import json
import email
import email.policy
from email.message import EmailMessage
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import extract_msg

    MSG_SUPPORT = True
except ImportError:
    MSG_SUPPORT = False
    logger.warning(
        "extract-msg library not found. MSG file support disabled. Install with: pip install extract-msg"
    )


class AttachmentExtractor:
    """Class for extracting attachments from MSG and EML files."""

    def __init__(self, output_folder: str = "attachments"):
        """
        Initialize the AttachmentExtractor.

        Args:
            output_folder (str): Default folder to save attachments
        """
        self.output_folder = output_folder
        self._ensure_output_folder()

    def _ensure_output_folder(self) -> None:
        """Create output folder if it doesn't exist."""
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def read_eml_file(self, file_path: str) -> EmailMessage:
        """
        Read an EML file and return the email message.

        Args:
            file_path (str): Path to the EML file

        Returns:
            EmailMessage: Parsed email message

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If the file cannot be parsed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EML file not found: {file_path}")

        try:
            with open(file_path, "rb") as f:
                msg = email.message_from_bytes(f.read(), policy=email.policy.default)
            logger.info(f"Successfully read EML file: {file_path}")
            return msg
        except Exception as e:
            logger.error(f"Error reading EML file {file_path}: {str(e)}")
            raise

    def read_msg_file(self, file_path: str):
        """
        Read an MSG file and return the message object.

        Args:
            file_path (str): Path to the MSG file

        Returns:
            extract_msg.Message: Parsed MSG message

        Raises:
            ImportError: If extract-msg library is not installed
            FileNotFoundError: If the file doesn't exist
            Exception: If the file cannot be parsed
        """
        if not MSG_SUPPORT:
            raise ImportError(
                "extract-msg library is required for MSG file support. Install with: pip install extract-msg"
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MSG file not found: {file_path}")

        try:
            msg = extract_msg.Message(file_path)
            logger.info(f"Successfully read MSG file: {file_path}")
            return msg
        except Exception as e:
            logger.error(f"Error reading MSG file {file_path}: {str(e)}")
            raise

    def extract_attachments_from_eml(
        self, file_path: str, output_folder: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract attachments from an EML file.

        Args:
            file_path (str): Path to the EML file
            output_folder (str, optional): Folder to save attachments

        Returns:
            List[Dict[str, str]]: List of attachment info dictionaries
        """
        output_folder = output_folder or self.output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        msg = self.read_eml_file(file_path)
        attachments = []

        for part in msg.walk():
            # Check if this part is an attachment
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    # Sanitize filename
                    filename = self._sanitize_filename(filename)
                    attachment_path = os.path.join(output_folder, filename)

                    # Save attachment
                    try:
                        with open(attachment_path, "wb") as f:
                            f.write(part.get_payload(decode=True))

                        attachment_info = {
                            "filename": filename,
                            "path": attachment_path,
                            "size": os.path.getsize(attachment_path),
                            "content_type": part.get_content_type(),
                        }
                        attachments.append(attachment_info)
                        logger.info(f"Saved attachment: {filename}")

                    except Exception as e:
                        logger.error(f"Error saving attachment {filename}: {str(e)}")

        return attachments

    def extract_attachments_from_msg(
        self, file_path: str, output_folder: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract attachments from an MSG file.

        Args:
            file_path (str): Path to the MSG file
            output_folder (str, optional): Folder to save attachments

        Returns:
            List[Dict[str, str]]: List of attachment info dictionaries
        """
        if not MSG_SUPPORT:
            raise ImportError(
                "extract-msg library is required for MSG file support. Install with: pip install extract-msg"
            )

        output_folder = output_folder or self.output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        msg = self.read_msg_file(file_path)
        attachments = []

        if hasattr(msg, "attachments") and msg.attachments:
            for attachment in msg.attachments:
                try:
                    filename = self._sanitize_filename(
                        attachment.longFilename
                        or attachment.shortFilename
                        or "unknown_attachment"
                    )
                    attachment_path = os.path.join(output_folder, filename)

                    # Save attachment
                    attachment.save(customPath=output_folder, customFilename=filename)

                    attachment_info = {
                        "filename": filename,
                        "path": attachment_path,
                        "size": (
                            os.path.getsize(attachment_path)
                            if os.path.exists(attachment_path)
                            else 0
                        ),
                        "content_type": getattr(
                            attachment, "mimeType", "application/octet-stream"
                        ),
                    }
                    attachments.append(attachment_info)
                    logger.info(f"Saved attachment: {filename}")

                except Exception as e:
                    logger.error(f"Error saving attachment: {str(e)}")

        return attachments

    def has_attachments(self, file_path: str) -> bool:
        """
        Check if a file (MSG or EML) has attachments.

        Args:
            file_path (str): Path to the email file

        Returns:
            bool: True if the file has attachments, False otherwise
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == ".eml":
                msg = self.read_eml_file(file_path)
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        return True
                return False

            elif file_ext == ".msg":
                if not MSG_SUPPORT:
                    logger.warning(
                        "Cannot check MSG attachments without extract-msg library"
                    )
                    return False

                msg = self.read_msg_file(file_path)
                return hasattr(msg, "attachments") and bool(msg.attachments)

            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return False

        except Exception as e:
            logger.error(f"Error checking attachments in {file_path}: {str(e)}")
            return False

    def process_email_file(
        self, file_path: str, output_folder: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process an email file (MSG or EML) and extract attachments if present.

        Args:
            file_path (str): Path to the email file
            output_folder (str, optional): Folder to save attachments

        Returns:
            Dict[str, any]: Processing results including attachment info
        """
        output_folder = output_folder or self.output_folder
        file_ext = os.path.splitext(file_path)[1].lower()

        result = {
            "file_path": file_path,
            "file_type": file_ext,
            "has_attachments": False,
            "attachments": [],
            "error": None,
        }

        try:
            # Check if file has attachments
            result["has_attachments"] = self.has_attachments(file_path)

            if result["has_attachments"]:
                # Extract attachments based on file type
                if file_ext == ".eml":
                    result["attachments"] = self.extract_attachments_from_eml(
                        file_path, output_folder
                    )
                elif file_ext == ".msg":
                    result["attachments"] = self.extract_attachments_from_msg(
                        file_path, output_folder
                    )

                logger.info(
                    f"Processed {file_path}: Found {len(result['attachments'])} attachments"
                )
            else:
                logger.info(f"Processed {file_path}: No attachments found")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {file_path}: {str(e)}")

        return result

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove invalid characters.

        Args:
            filename (str): Original filename

        Returns:
            str: Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove leading/trailing spaces and dots
        filename = filename.strip(" .")

        # Ensure filename is not empty
        if not filename:
            filename = "unnamed_attachment"

        return filename


# Convenience function to extract email body text from various file types
def extract_email_text(file_path: str) -> str:
    """
    Extract the textual body from an email file. Supports .msg, .eml, .md, .txt.

    Args:
        file_path (str): Absolute or relative path to the email file

    Returns:
        str: Extracted text content

    Raises:
        Exception: If the file cannot be read or parsed
    """
    try:
        suffix = Path(file_path).suffix.lower()

        # Plain text or markdown files
        if suffix in {".md", ".txt"}:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")

        # EML files
        if suffix == ".eml":
            extractor = AttachmentExtractor()
            msg = extractor.read_eml_file(file_path)

            # Prefer text/plain; fallback to text/html; finally raw payload
            try:
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True) or b""
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="ignore")

                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True) or b""
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="ignore")

                payload = msg.get_payload(decode=True) or b""
                if payload:
                    return payload.decode("utf-8", errors="ignore")
            except Exception:
                # Fall through to best-effort string representation
                pass

            return str(msg)

        # MSG files
        if suffix == ".msg":
            extractor = AttachmentExtractor()
            msg = extractor.read_msg_file(file_path)
            body = getattr(msg, "body", None) or getattr(msg, "htmlBody", None)
            if isinstance(body, bytes):
                try:
                    return body.decode("utf-8", errors="ignore")
                except Exception:
                    return body.decode(errors="ignore")
            return body or ""

        # Fallback: try reading as UTF-8 text
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to extract email text from {file_path}: {str(e)}")
        raise


def extract_text_from_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text content from a PDF file using pdfminer.six.

    Args:
        file_path (str): Path to the PDF file
        max_pages (Optional[int]): If provided, stop after reading this many pages

    Returns:
        str: Extracted text content or empty string if none
    """
    try:
        # Import locally to avoid heavy import unless needed
        from pdfminer.high_level import extract_text as pdf_extract_text

        if max_pages is None:
            return pdf_extract_text(file_path) or ""

        # To limit pages, pdfminer doesn't expose a simple arg; read all then slice by heuristic
        # Prefer full text for correctness; caller can trim length afterwards.
        return pdf_extract_text(file_path) or ""
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF '{file_path}': {str(e)}")
        return ""


def extract_email_metadata(file_path: str) -> Dict[str, any]:
    """
    Extract basic email metadata from .eml or .msg files. Returns empty fields for others.

    Fields: from, to, cc, bcc, subject, date, message_id, attachments_count, attachment_filenames
    """
    metadata: Dict[str, any] = {
        "from": None,
        "to": None,
        "cc": None,
        "bcc": None,
        "subject": None,
        "date": None,
        "message_id": None,
        "attachments_count": 0,
        "attachment_filenames": [],
    }

    try:
        suffix = Path(file_path).suffix.lower()

        if suffix == ".eml":
            msg = AttachmentExtractor().read_eml_file(file_path)
            metadata["from"] = msg.get("From")
            metadata["to"] = msg.get("To")
            metadata["cc"] = msg.get("Cc")
            metadata["bcc"] = msg.get("Bcc")
            metadata["subject"] = msg.get("Subject")
            metadata["date"] = msg.get("Date")
            metadata["message_id"] = msg.get("Message-ID")

            filenames: List[str] = []
            try:
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        fname = part.get_filename()
                        if fname:
                            filenames.append(fname)
            except Exception:
                pass
            metadata["attachment_filenames"] = filenames
            metadata["attachments_count"] = len(filenames)

        elif suffix == ".msg":
            if not MSG_SUPPORT:
                return metadata
            m = AttachmentExtractor().read_msg_file(file_path)
            # extract_msg exposes attributes; access defensively
            metadata["from"] = getattr(m, "sender", None)
            metadata["to"] = getattr(m, "to", None)
            metadata["cc"] = getattr(m, "cc", None)
            metadata["bcc"] = getattr(m, "bcc", None)
            metadata["subject"] = getattr(m, "subject", None)
            metadata["date"] = getattr(m, "date", None)
            metadata["message_id"] = getattr(m, "messageId", None)

            filenames: List[str] = []
            try:
                if hasattr(m, "attachments") and m.attachments:
                    for att in m.attachments:
                        fname = getattr(att, "longFilename", None) or getattr(
                            att, "shortFilename", None
                        )
                        if fname:
                            filenames.append(fname)
            except Exception:
                pass
            metadata["attachment_filenames"] = filenames
            metadata["attachments_count"] = len(filenames)

        # For .md/.txt or others, keep defaults

    except Exception as e:
        logger.warning(f"Failed to extract metadata from '{file_path}': {str(e)}")

    return metadata


def gather_attachments_text(
    email_file_path: str,
    output_folder: Optional[str] = None,
    max_chars_per_attachment: int = 8000,
) -> str:
    """
    Extract and aggregate text from text-like and PDF attachments found in an email file.

    Args:
        email_file_path (str): Path to .msg or .eml email file
        output_folder (Optional[str]): Folder to save extracted attachments
        max_chars_per_attachment (int): Max characters to include per attachment

    Returns:
        str: Bracketed sections per attachment or empty string if none.
             Format per attachment:
             [ATTACHMENT name="..." content_type="..." size="..."]\n
             ...extracted text...\n
             [/ATTACHMENT]
    """
    try:
        # Use convenience function which ensures output folder exists
        output_folder = output_folder or "attachments"
        extraction_result = extract_attachments(email_file_path, output_folder)

        if not extraction_result or not extraction_result.get("attachments"):
            return ""

        attachments_text_parts: List[str] = []
        attachments_index: List[Dict[str, any]] = []
        for att in extraction_result.get("attachments", []):
            fname = att.get("filename")
            ctype = (att.get("content_type") or "").lower()
            apath = att.get("path")
            asize = att.get("size")

            if not apath or not Path(apath).exists():
                continue

            ext = Path(str(fname or "")).suffix.lower()

            text_content = ""
            is_text_like = bool(ctype.startswith("text/")) or ext in {
                ".txt",
                ".csv",
                ".md",
                ".json",
                ".xml",
                ".html",
                ".htm",
            }

            if ext == ".pdf" or ctype == "application/pdf":
                text_content = extract_text_from_pdf(apath)
            elif is_text_like:
                try:
                    text_content = Path(apath).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed reading text attachment '{fname}': {str(e)}"
                    )

            # Always record in index (even if no extractable text)
            try:
                attachments_index.append(
                    {
                        "name": fname,
                        "content_type": ctype,
                        "size": asize,
                    }
                )
            except Exception:
                pass

            if text_content:
                header = f'[ATTACHMENT name="{fname}" content_type="{ctype}" size="{asize}"]\n'
                body = (text_content[:max_chars_per_attachment] or "").strip()
                footer = "\n[/ATTACHMENT]"
                attachments_text_parts.append(header + body + footer)

            # Remove the saved attachment file after processing
            try:
                Path(apath).unlink(missing_ok=True)
            except Exception as del_exc:
                logger.warning(
                    f"Failed to delete temporary attachment '{apath}': {str(del_exc)}"
                )

        if not attachments_text_parts and not attachments_index:
            return ""

        index_block = ""
        if attachments_index:
            try:
                index_json = json.dumps(attachments_index, ensure_ascii=False)
            except Exception:
                index_json = "[]"
            index_block = (
                "[ATTACHMENTS_INDEX]\n" + index_json + "\n[/ATTACHMENTS_INDEX]"
            )

        # Join with a blank line between blocks; no markdown headers to keep it simple
        blocks = [b for b in [index_block, "\n\n".join(attachments_text_parts)] if b]
        return "\n\n".join(blocks)
    except Exception as e:
        logger.warning(
            f"Failed extracting and aggregating text from attachments in '{email_file_path}': {str(e)}"
        )
        return ""


# Convenience functions for direct use
def extract_attachments(
    file_path: str, output_folder: str = "attachments"
) -> Dict[str, any]:
    """
    Convenience function to extract attachments from an email file.

    Args:
        file_path (str): Path to the email file (MSG or EML)
        output_folder (str): Folder to save attachments

    Returns:
        Dict[str, any]: Processing results
    """
    extractor = AttachmentExtractor(output_folder)
    return extractor.process_email_file(file_path, output_folder)


def check_for_attachments(file_path: str) -> bool:
    """
    Convenience function to check if an email file has attachments.

    Args:
        file_path (str): Path to the email file (MSG or EML)

    Returns:
        bool: True if the file has attachments, False otherwise
    """
    extractor = AttachmentExtractor()
    return extractor.has_attachments(file_path)


# Example usage
if __name__ == "__main__":
    # Example usage
    extractor = AttachmentExtractor("extracted_attachments")

    # Process a single file
    result = extractor.process_email_file("example.eml")
    print(f"Processing result: {result}")

    # Or use convenience functions
    has_attachments = check_for_attachments("example.eml")
    if has_attachments:
        attachments = extract_attachments("example.eml", "my_attachments")
        print(f"Extracted {len(attachments['attachments'])} attachments")

# test = AttachmentExtractor("/Users/I565406/Projects/BTP AI CoE/POCs/ai-powered-email-cockpit/api/app/utils")
# test.extract_attachments_from_msg("/Users/I565406/Downloads/Advance Notification letter for Acc# 0007068206.msg")
