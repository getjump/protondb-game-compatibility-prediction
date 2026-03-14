"""Pydantic response schemas for LLM structured outputs.

These models serve dual purpose:
1. Generate JSON Schema for structured outputs (response_format)
2. Post-validate LLM responses when structured outputs aren't available
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- GPU Normalization ---

class GpuNormResult(BaseModel):
    vendor: Literal["nvidia", "amd", "intel", "unknown"] = "unknown"
    family: str = "unknown"
    model: str = "unknown"
    normalized_name: str = "unknown"
    is_apu: bool = False
    is_igpu: bool = False
    is_virtual: bool = False


class GpuNormBatchResponse(BaseModel):
    results: list[GpuNormResult]


# --- CPU Normalization ---

class CpuNormResult(BaseModel):
    vendor: Literal["intel", "amd", "unknown"] = "unknown"
    family: str = "unknown"
    model: str = "unknown"
    normalized_name: str = "unknown"
    generation: int | None = None
    is_apu: bool = False


class CpuNormBatchResponse(BaseModel):
    results: list[CpuNormResult]


# --- Launch Options Parsing ---

class EnvVar(BaseModel):
    name: str
    value: str


class Wrapper(BaseModel):
    tool: Literal["gamescope", "mangohud", "gamemoderun", "prime-run", "taskset", "obs-gamecapture", "other"] = "other"
    args: str = ""


class LaunchParseResult(BaseModel):
    env_vars: list[EnvVar] = Field(default_factory=list)
    wrappers: list[Wrapper] = Field(default_factory=list)
    game_args: list[str] = Field(default_factory=list)
    unparsed: str = ""


class LaunchParseBatchResponse(BaseModel):
    results: list[LaunchParseResult]
