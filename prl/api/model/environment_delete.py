from pydantic import BaseModel, Field


class EnvironmentDeletion(BaseModel):
    success: bool = Field(
        ...,
        example=True,
        description="True if the environment with given id "
                    "was deleted successfully, false otherwise."
    )
