from fastapi import APIRouter, Depends
from ..security import get_api_key

router = APIRouter(
    prefix="/service1",
    tags=["service1"],
    dependencies=[Depends(get_api_key)],
)


@router.get("/")
def read_service1():
    return {"message": "This is a service1 route."}
