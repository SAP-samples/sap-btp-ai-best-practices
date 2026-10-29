"""Complete-turn HANA snapshots and idempotent A2A request claims."""
import json
import time
from ..services.workspace.schema import transaction,decode_json
from ..models.workspace import RevisionConflict


class ConversationStore:
    """Persist complete LangChain messages and cached responses within an access domain."""

    def __init__(self,backend,db_path=None):
        """Create required tables automatically; SQLite requires explicit test injection."""
        self.backend,self.db_path=backend,db_path
        with backend.get_connection(db_path) as connection:
            cursor=backend.cursor(connection);text='NCLOB' if backend.is_hana else 'TEXT'
            schemas={
                'RECEIVABLES_CONVERSATIONS':f'CREATE TABLE RECEIVABLES_CONVERSATIONS (context_id NVARCHAR(128) PRIMARY KEY, owner_id NVARCHAR(128), revision INTEGER, active_request NVARCHAR(128), claimed_at DOUBLE, payload {text})',
                'RECEIVABLES_AGENT_REQUESTS':f'CREATE TABLE RECEIVABLES_AGENT_REQUESTS (context_id NVARCHAR(128), request_id NVARCHAR(128), input_hash NVARCHAR(64), status NVARCHAR(32), task_id NVARCHAR(128), response {text}, PRIMARY KEY (context_id,request_id))'}
            for name,ddl in schemas.items():
                if not backend.table_exists(connection,name):cursor.execute(ddl)
            backend.commit(connection)

    def load(self,context_id,owner_id):
        """Read the last complete turn only after matching its authenticated access domain."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT owner_id,revision,payload FROM RECEIVABLES_CONVERSATIONS WHERE context_id = ?',(context_id,))
            row=cursor.fetchone()
            if row is None:return {'context_id':context_id,'revision':0,'messages':[],'history':[]}
            if row[0]!=owner_id:raise LookupError('Conversation not found')
            return {**decode_json(row[2]),'revision':row[1]}

    def claim(self,context_id,request_id,owner_id,input_hash,expected_revision=None):
        """Claim one turn atomically or return its completed cached response without work."""
        if any(not isinstance(value,str) or not value or len(value)>128 for value in (context_id,request_id,owner_id)):
            raise ValueError('Conversation and request IDs must be 1–128 characters')
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT owner_id,revision,active_request,claimed_at,payload FROM RECEIVABLES_CONVERSATIONS WHERE context_id = ?',(context_id,))
            row=cursor.fetchone()
            if row is None:
                initial={'context_id':context_id,'messages':[],'history':[]}
                cursor.execute('INSERT INTO RECEIVABLES_CONVERSATIONS (context_id,owner_id,revision,payload) VALUES (?, ?, 0, ?)',(context_id,owner_id,json.dumps(initial)))
                revision,active,claimed,payload=0,None,None,initial
            else:
                if row[0]!=owner_id:raise LookupError('Conversation not found')
                revision,active,claimed,payload=row[1],row[2],row[3],decode_json(row[4])
            cursor.execute('SELECT input_hash,status,response FROM RECEIVABLES_AGENT_REQUESTS WHERE context_id = ? AND request_id = ?',(context_id,request_id))
            previous=cursor.fetchone()
            if previous:
                if previous[0]!=input_hash:raise RevisionConflict('Request ID already identifies different input')
                if previous[1]=='completed':return {'cached':decode_json(previous[2])}
                raise RevisionConflict('This request is active or interrupted; use a new request ID for an explicit retry')
            # An invocation is bounded to 180 seconds. A ten-minute abandoned claim can
            # be interrupted without replaying any partially completed tool sequence.
            if active and time.time()-float(claimed or 0)>600:
                cursor.execute("UPDATE RECEIVABLES_AGENT_REQUESTS SET status = 'interrupted' WHERE context_id = ? AND request_id = ?",(context_id,active))
                cursor.execute('UPDATE RECEIVABLES_CONVERSATIONS SET active_request = NULL WHERE context_id = ? AND active_request = ?',(context_id,active))
                active=None
            if active:raise RevisionConflict('Another turn is active in this conversation; retry after it completes')
            if expected_revision is not None and revision!=expected_revision:raise RevisionConflict('Conversation revision changed')
            cursor.execute('UPDATE RECEIVABLES_CONVERSATIONS SET active_request = ?,claimed_at = ? WHERE context_id = ? AND revision = ? AND active_request IS NULL',
                           (request_id,time.time(),context_id,revision))
            if cursor.rowcount!=1:raise RevisionConflict('Another turn claimed this conversation')
            cursor.execute("INSERT INTO RECEIVABLES_AGENT_REQUESTS (context_id,request_id,input_hash,status) VALUES (?, ?, ?, 'running')",(context_id,request_id,input_hash))
            return {**payload,'revision':revision}

    def complete(self,context_id,request_id,owner_id,expected_revision,messages,response,history,workspace_context):
        """Atomically publish a whole successful turn and its reusable A2A response."""
        payload=dict(context_id=context_id,messages=messages,history=history,workspace_context=workspace_context)
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('UPDATE RECEIVABLES_CONVERSATIONS SET revision = ?,active_request = NULL,payload = ? WHERE context_id = ? AND owner_id = ? AND revision = ? AND active_request = ?',
                (expected_revision+1,json.dumps(payload),context_id,owner_id,expected_revision,request_id))
            if cursor.rowcount!=1:raise RevisionConflict('Conversation claim changed before completion')
            cursor.execute("UPDATE RECEIVABLES_AGENT_REQUESTS SET status = 'completed',response = ?,task_id = ? WHERE context_id = ? AND request_id = ?",
                (json.dumps(response),response.get('id'),context_id,request_id))
        return response

    def fail(self,context_id,request_id,owner_id):
        """Release a failed claim while preserving the previous complete conversation."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('UPDATE RECEIVABLES_CONVERSATIONS SET active_request = NULL WHERE context_id = ? AND owner_id = ? AND active_request = ?',
                           (context_id,owner_id,request_id))
            if cursor.rowcount:
                cursor.execute("UPDATE RECEIVABLES_AGENT_REQUESTS SET status = 'interrupted' WHERE context_id = ? AND request_id = ?",(context_id,request_id))

    def task(self,task_id,owner_id):
        """Retrieve completed A2A tasks after restart under the same access domain."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT r.response FROM RECEIVABLES_AGENT_REQUESTS r JOIN RECEIVABLES_CONVERSATIONS c ON r.context_id = c.context_id WHERE r.task_id = ? AND c.owner_id = ?', (task_id,owner_id))
            row=cursor.fetchone()
            if row is None:raise LookupError('Task not found')
            return decode_json(row[0])
