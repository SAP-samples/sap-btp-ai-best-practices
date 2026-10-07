"""Explicit developer experiment routes and narrowly scoped CF-task callbacks."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.security import get_api_key
from app.workspace.dependencies import get_service
from production_wheel.constraints.code_experiment import MAX_INPUT_BYTES
from production_wheel.constraints.code_experiment_cf import ConstraintCodeBroker, decode

router = APIRouter(prefix="/api/constraint-code/jobs", tags=["constraint-code-experiment"])


def get_broker(service=Depends(get_service)):
    """Use the existing HANA service repository for durable job and artifact storage."""
    return ConstraintCodeBroker(service.repo)


async def body(request):
    """Read a bounded JSON request without eagerly buffering an oversized body."""
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > MAX_INPUT_BYTES:
            raise HTTPException(413, "experiment input limit exceeded")
    try:
        return decode(chunks)
    except (ValueError, RecursionError) as error:
        raise HTTPException(422, "invalid JSON") from error


def capability(request):
    """Extract only a job-scoped bearer capability; never accept the application API key."""
    prefix, separator, token = request.headers.get("Authorization", "").partition(" ")
    if prefix != "Bearer" or not separator or not 20 <= len(token) <= 128:
        raise HTTPException(403, "job capability required")
    return token


@router.post("", dependencies=[Depends(get_api_key)])
async def create_job(request: Request, broker=Depends(get_broker)):
    """Queue an explicit authenticated developer experiment; no automatic task launch."""
    packet = await body(request)
    if not isinstance(packet, dict) or not {"code", "payload"} <= packet.keys() or packet.keys() - {"code", "payload", "timeout_seconds"}:
        raise HTTPException(422, "expected code, payload and optional timeout_seconds")
    try:
        return broker.create(**packet)
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/{identifier}", dependencies=[Depends(get_api_key)])
def inspect_job(identifier: str, broker=Depends(get_broker)):
    """Return lifecycle and disposed-task output to authenticated developer callers."""
    try:
        return broker.inspect(identifier)
    except KeyError as error:
        raise HTTPException(404, "unknown job") from error


@router.get("/{identifier}/input")
def fetch_input(identifier: str, request: Request, broker=Depends(get_broker)):
    """Deliver one input to its temporary capability holder and reject replay."""
    token = capability(request)
    try:
        return broker.input(identifier, token)
    except (KeyError, ValueError) as error:
        raise HTTPException(403, "invalid, expired or consumed job capability") from error


@router.post("/{identifier}/result")
async def post_result(identifier: str, request: Request, broker=Depends(get_broker)):
    """Accept one strictly validated callback using only its per-job capability."""
    token = capability(request)
    result = await body(request)
    try:
        broker.complete(identifier, token, result)
        return {"accepted": True}
    except (KeyError, ValueError) as error:
        raise HTTPException(403, "invalid capability or experiment result") from error
