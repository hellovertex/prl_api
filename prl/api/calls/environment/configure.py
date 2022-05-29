import logging

from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_config import EnvironmentConfig, EnvironmentConfigRequestBody

logger = logging.getLogger('configure')

router = APIRouter()


@router.post("/environment/configure",
             response_model=EnvironmentConfig,
             operation_id="configure_environment")
async def configure_environment(body: EnvironmentConfigRequestBody, request: Request):
    # request: Request, n_players: int, starting_stack_size: int
    """Creates an environment in the backend and returns its unique ID.
    Use this ID with /reset and /step endpoints to play the game.

    Internal: Calls backend.EnvironmentRegistry.add_environment(...),
    returns its id and config wrapped in EnvironmentConfig Model class"""
    # logger.warning(f'Request = {request.query_params}')
    # logger.warning(f'n_players = {body.n_players}')
    # logger.warning(f'starting_stack_size = {body.starting_stack_size}')
    n_players = body.n_players
    starting_stack_size = body.starting_stack_size
    assert 2 <= n_players <= 6
    # make args for env
    print(n_players)
    print(starting_stack_size)
    config = {"n_players": n_players,
              "starting_stack_size": starting_stack_size}

    return EnvironmentConfig(env_id=request.app.backend.add_environment(config),
                             num_players=n_players,
                             starting_stack_size=starting_stack_size)
