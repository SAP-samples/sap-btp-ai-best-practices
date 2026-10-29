"""Customer onboarding tools exposed only to the separate Rules Assistant."""
import re
from contextvars import ContextVar
from langchain_core.tools import tool
from sqlalchemy import text
from ...payment_advice.customers import normalize_client_key, get_customer, set_critical
from ...payment_advice.hana_schema import CUSTOMERS

current_request = ContextVar('customer_admin_request', default='')


def build_customer_admin_tools(engine):
    """Bind validated customer creation/priority tools to HANA and the current user turn."""
    @tool
    def create_customer(display_name: str, priority: bool = False) -> dict:
        """Create a customer only on explicit user request; priority means a dedicated extraction schema."""
        request = current_request.get()
        if not re.search(r'\b(create|add|register|onboard|crea|añade|alta)\b', request, re.I) or display_name.casefold() not in request.casefold():
            raise ValueError('Explicit current user request naming the new customer is required')
        key = normalize_client_key(display_name)
        if len(key) > 60 or len(display_name) > 200:
            raise ValueError('Customer name exceeds registry limits')
        if priority and not re.search(r'\b(priority|critical|prioritario|prioridad)\b', request, re.I):
            raise ValueError('Explicit priority request is required')
        if get_customer(engine, key):
            raise ValueError('Customer exists; use set_customer_priority for a priority change')
        with engine.begin() as conn:
            conn.execute(text(f'''INSERT INTO "{CUSTOMERS}" ("CLIENT_KEY","DISPLAY_NAME","IS_CRITICAL","STATUS")
                VALUES (:key,:name,:priority,'active')'''), {'key': key, 'name': display_name.strip(), 'priority': priority})
        return {'client_key': key, 'display_name': display_name.strip(), 'priority': priority}

    @tool
    def set_customer_priority(client_key: str, priority: bool) -> dict:
        """Set priority only on explicit current user request naming this existing customer."""
        request = current_request.get()
        customer = get_customer(engine, client_key)
        if not customer:
            raise ValueError('Unknown customer')
        if not re.search(r'\b(set|make|promote|demote|change|mark|cambia|marca)\b', request, re.I) or not re.search(r'\b(priority|critical|prioritario|prioridad)\b', request, re.I):
            raise ValueError('Explicit priority change request is required')
        if customer.display_name.casefold() not in request.casefold() and client_key.casefold() not in request.casefold():
            raise ValueError('The current request must identify the customer')
        set_critical(engine, client_key, priority)
        return {'client_key': client_key, 'priority': priority}

    return [create_customer, set_customer_priority]
