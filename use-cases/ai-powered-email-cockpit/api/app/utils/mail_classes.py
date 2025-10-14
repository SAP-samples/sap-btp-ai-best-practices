import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Sender:
    """Represents the email sender information"""
    name: str
    email: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sender':
        """Create Sender from dictionary"""
        return cls(
            name=data.get('name', ''),
            email=data.get('email', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Sender to dictionary"""
        return {
            'name': self.name,
            'email': self.email
        }


@dataclass
class EmailBody:
    """Represents the email body content"""
    text: str
    html: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailBody':
        """Create EmailBody from dictionary"""
        return cls(
            text=data.get('text', ''),
            html=data.get('html', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EmailBody to dictionary"""
        return {
            'text': self.text,
            'html': self.html
        }


@dataclass
class Email:
    """Main email class containing all email data"""
    sender: Sender
    subject: str
    body: EmailBody
    sent_date: str
    message_id: str
    tags: List[str] = field(default_factory=list)
    status: str = ""
    priority: str = ""
    thread_id: str = ""
    folder: str = ""
    
    @classmethod
    def from_json(cls, json_string: str) -> 'Email':
        """Create Email from JSON string"""
        data = json.loads(json_string)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Email':
        """Create Email from dictionary"""
        return cls(
            sender=Sender.from_dict(data.get('from', {})),
            subject=data.get('subject', ''),
            body=EmailBody.from_dict(data.get('body', {})),
            sent_date=data.get('sentDate', ''),
            message_id=data.get('messageId', ''),
            tags=data.get('tags', []),
            status=data.get('status', ''),
            priority=data.get('priority', ''),
            thread_id=data.get('thread_id', ''),
            folder=data.get('folder', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Email to dictionary"""
        return {
            'from': self.sender.to_dict(),
            'subject': self.subject,
            'body': self.body.to_dict(),
            'sentDate': self.sent_date,
            'messageId': self.message_id,
            'tags': self.tags,
            'status': self.status,
            'priority': self.priority,
            'thread_id': self.thread_id,
            'folder': self.folder
        }
    
    def to_json(self) -> str:
        """Convert Email to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def sent_datetime(self) -> Optional[datetime]:
        """Parse sent_date string into datetime object"""
        try:
            return datetime.fromisoformat(self.sent_date.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the email"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the email"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def set_status(self, status: str) -> None:
        """Set the email status"""
        self.status = status
    
    def set_priority(self, priority: str) -> None:
        """Set the email priority"""
        self.priority = priority
