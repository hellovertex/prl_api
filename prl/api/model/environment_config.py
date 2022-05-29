from typing import Optional

from pydantic import BaseModel, Field


class EnvironmentConfigRequestBody(BaseModel):
    n_players: int
    starting_stack_size: int

    class Config:
        schema_extra = {
            "n_players": {
                "example": 6,
                "description": "The environments number of starting players. 2 <= num_players <= 6"
            },
            "starting_stack_size": {
                "example": 20000,
                "description": "The number of chips each player will get on resetting the environment."
            }
        }


class EnvironmentConfig(BaseModel):
    env_id: Optional[int] = Field(
        ...,
        example=1,
        description="The environment unique id "
                    "used for requesting this specific environment."
    )
    num_players: int = Field(
        ...,
        example=6,
        description="The environments number of starting players. 2 <= num_players <= 6"
    )
    starting_stack_size: int = Field(
        ...,
        example=20000,
        description="The number of chips each player will get on resetting the environment."
    )
