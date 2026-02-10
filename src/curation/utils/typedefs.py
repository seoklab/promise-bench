from dataclasses import dataclass

import numpy as np
from pydantic import AliasPath, BaseModel, ConfigDict, Field, field_validator


class LooseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class GroupSet(LooseModel):
    center: str = Field(validation_alias=AliasPath("representative", "identifier"))
    members: list[str] = Field(validation_alias="result_set")

    @field_validator("members", mode="before")
    @staticmethod
    def _extract_members(v):
        try:
            return [d["identifier"] for d in v]
        except TypeError:
            return v


@dataclass
class TMScoreResult:
    chains: np.ndarray
    tm_scores: np.ndarray
