from fastapi import APIRouter

router = APIRouter()

@router.get("/explain")
def explain_stub():
    return {
        "status": "not_implemented",
        "message": "Explainability is scaffolded but not required for this submission."
    }
