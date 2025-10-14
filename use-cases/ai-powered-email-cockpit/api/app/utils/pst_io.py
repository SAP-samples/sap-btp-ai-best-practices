#!/usr/bin/env python3
"""
PST File Reader/Writer Module

This module provides functions to read from and write to PST (Personal Storage Table) files
used by Microsoft Outlook. It uses the pypff library for reading PST files and formats
the output according to a specific JSON structure.

Requirements:
    pip install pypff-python
    
Note: Writing to PST files is complex and typically requires specialized libraries.
This example shows the structure for reading and basic data extraction.
"""

import pypff
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSTReader:
    """Class to handle reading PST files."""
    
    def __init__(self, pst_path: str):
        """Initialize PST reader with file path."""
        self.pst_path = pst_path
        self.pst_file = None
        
    def open(self) -> bool:
        """Open the PST file for reading."""
        try:
            self.pst_file = pypff.file()
            self.pst_file.open(self.pst_path)
            logger.info(f"Successfully opened PST file: {self.pst_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open PST file: {e}")
            return False
            
    def close(self):
        """Close the PST file."""
        if self.pst_file:
            self.pst_file.close()
            logger.info("PST file closed")
            
    def get_folder_structure(self) -> Dict:
        """Get the folder structure of the PST file."""
        if not self.pst_file:
            raise ValueError("PST file not opened")
            
        def traverse_folder(folder, level=0):
            folder_info = {
                'name': folder.name,
                'level': level,
                'item_count': folder.number_of_sub_messages,
                'subfolders': []
            }
            
            # Traverse subfolders
            for subfolder in folder.sub_folders:
                folder_info['subfolders'].append(traverse_folder(subfolder, level + 1))
                
            return folder_info
            
        root_folder = self.pst_file.root_folder
        return traverse_folder(root_folder)
    
    def _extract_message_id(self, message) -> str:
        """Extract message ID from email."""
        try:
            # Try to get the Message-ID header
            if hasattr(message, 'message_id') and message.message_id:
                return message.message_id
            elif hasattr(message, 'internet_message_id') and message.internet_message_id:
                return message.internet_message_id
            else:
                # Generate a fallback message ID if none exists
                timestamp = message.creation_time or datetime.now()
                return f"<generated_{timestamp.strftime('%Y%m%d_%H%M%S')}@pst.reader>"
        except Exception:
            return f"<unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}@pst.reader>"
    
    def _format_datetime(self, dt) -> str:
        """Format datetime to ISO 8601 string."""
        if dt is None:
            return ""
        try:
            # Ensure the datetime is timezone-aware (add UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return dt.isoformat().replace('+00:00', 'Z')
        except Exception:
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Remove null bytes and normalize line endings
        text = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        return text.strip()
    
    def _extract_thread_id(self, message) -> str:
        """Extract thread ID from message."""
        try:
            # Try various thread-related properties
            if hasattr(message, 'conversation_id') and message.conversation_id:
                return str(message.conversation_id)
            elif hasattr(message, 'thread_id') and message.thread_id:
                return str(message.thread_id)
            elif hasattr(message, 'conversation_topic') and message.conversation_topic:
                # Generate thread ID based on conversation topic
                topic = self._clean_text(message.conversation_topic)
                return f"thread_{hash(topic) % 100000000}"
            else:
                return ""
        except Exception:
            return ""
    
    def _get_priority(self, message) -> str:
        """Extract message priority."""
        try:
            if hasattr(message, 'priority'):
                priority_map = {
                    0: "low",
                    1: "normal", 
                    2: "high"
                }
                return priority_map.get(message.priority, "")
            return ""
        except Exception:
            return ""
    
    def _get_status(self, message) -> str:
        """Extract message status/flags."""
        try:
            status_flags = []
            if hasattr(message, 'is_read') and message.is_read:
                status_flags.append("read")
            if hasattr(message, 'is_replied') and message.is_replied:
                status_flags.append("replied")
            if hasattr(message, 'is_forwarded') and message.is_forwarded:
                status_flags.append("forwarded")
            return ",".join(status_flags)
        except Exception:
            return ""
        
    def extract_emails(self, folder=None, limit: Optional[int] = None) -> List[Dict]:
        """Extract emails from PST file in the specified JSON format."""
        if not self.pst_file:
            raise ValueError("PST file not opened")
            
        emails = []
        
        def process_folder(current_folder):
            nonlocal emails
            
            # Process messages in current folder
            for message in current_folder.sub_messages:
                if limit and len(emails) >= limit:
                    return
                    
                try:
                    # Extract sender information
                    sender_name = self._clean_text(message.sender_name or "")
                    sender_email = self._clean_text(message.sender_email_address or "")
                    
                    # Extract body content
                    text_body = self._clean_text(message.plain_text_body or "")
                    html_body = self._clean_text(message.html_body or "")
                    
                    # Create email data structure matching the required format
                    email_data = {
                        "from": {
                            "name": sender_name,
                            "email": sender_email
                        },
                        "subject": self._clean_text(message.subject or ""),
                        "body": {
                            "text": text_body,
                            "html": html_body
                        },
                        "sentDate": self._format_datetime(message.client_submit_time or message.delivery_time),
                        "messageId": self._extract_message_id(message),
                        "tags": [],  # PST files don't typically have tags, but we keep the structure
                        "status": self._get_status(message),
                        "priority": self._get_priority(message),
                        "thread_id": self._extract_thread_id(message),
                        "folder": current_folder.name or ""
                    }
                    
                    # Add recipients information (optional extension)
                    recipients = []
                    try:
                        for recipient in message.recipients:
                            recipients.append({
                                "name": self._clean_text(recipient.name or ""),
                                "email": self._clean_text(recipient.email_address or ""),
                                "type": getattr(recipient, 'type', 'to')
                            })
                    except Exception:
                        pass
                    
                    # Add recipients to email data (not in original structure but useful)
                    if recipients:
                        email_data["recipients"] = recipients
                    
                    # Add attachment information (optional extension)
                    attachments = []
                    try:
                        for attachment in message.attachments:
                            attachments.append({
                                "filename": self._clean_text(attachment.name or "unnamed"),
                                "size": getattr(attachment, 'size', 0)
                            })
                    except Exception:
                        pass
                        
                    if attachments:
                        email_data["attachments"] = attachments
                    
                    emails.append(email_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")
                    continue
            
            # Process subfolders
            for subfolder in current_folder.sub_folders:
                process_folder(subfolder)
        
        start_folder = folder if folder else self.pst_file.root_folder
        process_folder(start_folder)
        
        return emails
    
    def extract_contacts(self) -> List[Dict]:
        """Extract contacts from PST file."""
        if not self.pst_file:
            raise ValueError("PST file not opened")
            
        contacts = []
        
        def process_folder(folder):
            for message in folder.sub_messages:
                # Check if this is a contact item
                if hasattr(message, 'message_class') and 'IPM.Contact' in str(message.message_class):
                    try:
                        contact = {
                            'display_name': self._clean_text(getattr(message, 'display_name', '')),
                            'email': self._clean_text(getattr(message, 'email_address', '')),
                            'company': self._clean_text(getattr(message, 'company_name', '')),
                            'phone': self._clean_text(getattr(message, 'business_telephone_number', '')),
                            'mobile': self._clean_text(getattr(message, 'mobile_telephone_number', '')),
                            'address': self._clean_text(getattr(message, 'business_address', ''))
                        }
                        contacts.append(contact)
                    except Exception as e:
                        logger.warning(f"Error processing contact: {e}")
                        
            # Process subfolders
            for subfolder in folder.sub_folders:
                process_folder(subfolder)
                
        process_folder(self.pst_file.root_folder)
        return contacts


class PSTWriter:
    """
    Class to handle writing PST-extracted data to various formats.
    Note: Direct PST writing is complex and typically requires commercial libraries.
    This class demonstrates exporting data in the specified JSON structure.
    """
    
    def __init__(self, output_path: str):
        """Initialize PST writer with output path."""
        self.output_path = output_path
        
    def save_emails_json(self, emails: List[Dict], pretty_print: bool = True):
        """Save emails to JSON file in the specified structure."""
        json_path = self.output_path.replace('.pst', '_emails.json')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(emails, f, indent=2, ensure_ascii=False)
            else:
                json.dump(emails, f, ensure_ascii=False)
                
        logger.info(f"Emails exported to JSON: {json_path}")
        return json_path
    
    def save_emails_csv(self, emails: List[Dict]):
        """Export emails to CSV format."""
        if not emails:
            return None
            
        csv_path = self.output_path.replace('.pst', '_emails.csv')
        fieldnames = [
            'sender_name', 'sender_email', 'subject', 'sentDate', 'messageId',
            'text_body', 'html_body', 'status', 'priority', 'thread_id', 'folder'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for email in emails:
                row = {
                    'sender_name': email.get('from', {}).get('name', ''),
                    'sender_email': email.get('from', {}).get('email', ''),
                    'subject': email.get('subject', ''),
                    'sentDate': email.get('sentDate', ''),
                    'messageId': email.get('messageId', ''),
                    'text_body': email.get('body', {}).get('text', '')[:2000],  # Truncate for CSV
                    'html_body': email.get('body', {}).get('html', '')[:2000],  # Truncate for CSV
                    'status': email.get('status', ''),
                    'priority': email.get('priority', ''),
                    'thread_id': email.get('thread_id', ''),
                    'folder': email.get('folder', '')
                }
                writer.writerow(row)
                
        logger.info(f"Emails exported to CSV: {csv_path}")
        return csv_path
    
    def save_complete_export(self, emails: List[Dict], contacts: List[Dict] = None):
        """Save complete export with metadata."""
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "email_count": len(emails),
                "contact_count": len(contacts or []),
                "format_version": "1.0"
            },
            "emails": emails,
            "contacts": contacts or []
        }
        
        json_path = self.output_path.replace('.pst', '_complete_export.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Complete export saved to: {json_path}")
        return json_path


def read_pst_file(pst_path: str, extract_emails: bool = True, extract_contacts: bool = True, 
                  email_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to read PST file and extract data in the specified format.
    
    Args:
        pst_path: Path to PST file
        extract_emails: Whether to extract emails
        extract_contacts: Whether to extract contacts
        email_limit: Maximum number of emails to extract
        
    Returns:
        Dictionary containing extracted data
    """
    reader = PSTReader(pst_path)
    
    if not reader.open():
        return {}
        
    try:
        result = {
            'folder_structure': reader.get_folder_structure(),
            'emails': [],
            'contacts': []
        }
        
        if extract_emails:
            result['emails'] = reader.extract_emails(limit=email_limit)
            
        if extract_contacts:
            result['contacts'] = reader.extract_contacts()
            
        return result
        
    finally:
        reader.close()


def save_emails_to_json(emails: List[Dict], output_path: str, pretty_print: bool = True) -> str:
    """
    Save emails list directly to JSON file in the specified format.
    
    Args:
        emails: List of email dictionaries in the specified format
        output_path: Path for output JSON file
        pretty_print: Whether to format JSON with indentation
        
    Returns:
        Path to saved JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(emails, f, indent=2, ensure_ascii=False)
        else:
            json.dump(emails, f, ensure_ascii=False)
    
    logger.info(f"Emails saved to: {output_path}")
    return output_path


def convert_pst_to_json(pst_path: str, output_path: str, email_limit: Optional[int] = None) -> str:
    """
    Convert PST file directly to JSON format.
    
    Args:
        pst_path: Path to input PST file
        output_path: Path for output JSON file
        email_limit: Maximum number of emails to extract
        
    Returns:
        Path to output JSON file
    """
    data = read_pst_file(pst_path, email_limit=email_limit)
    
    if data and data.get('emails'):
        return save_emails_to_json(data['emails'], output_path)
    else:
        logger.error("No email data extracted from PST file")
        return ""
