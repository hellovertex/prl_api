from typing import Optional

from pydantic import BaseModel


class Stacks(BaseModel):
    stack_p0: Optional[int]
    stack_p1: Optional[int]
    stack_p2: Optional[int]
    stack_p3: Optional[int]
    stack_p4: Optional[int]
    stack_p5: Optional[int]


class EnvironmentResetRequestBody(BaseModel):
    env_id: int
    stack_sizes: Optional[Stacks]

    class Config:
        schema_extra = {
            "env_id": {
                "example": 1,
                "description": "The environment unique id "
                               "used for requesting this specific environment."
            },
            "stack_sizes": {
                "example": 20000,
                "description": "The number of chips each player will get on resetting the environment."
                               "Note that the environment is reset on each hand dealt. This implies"
                               "that starting stacks can vary between players, e.g. in the middle of the game."
            }
        }
