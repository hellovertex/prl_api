from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_delete import EnvironmentDeletion

router = APIRouter()


@router.get("/environment/{env_id}/delete",
            response_model=EnvironmentDeletion,
            operation_id="step_environment")
async def delete_environment(request: Request,
                             env_id: int, ):
    del request.app.backend.active_ens[env_id]
    success = False
    try:
        _ = request.app.backend.active_ens[env_id]
    except KeyError:
        success = True

    return {'success': success}
