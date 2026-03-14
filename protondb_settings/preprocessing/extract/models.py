"""Pydantic models for text extraction results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "env_var",
    "game_arg",
    "wrapper_config",
    "runner_selection",
    "protontricks_verb",
    "dll_override",
    "prefix_action",
    "file_patch",
    "registry_patch",
    "executable_override",
    "dependency_install",
    "session_requirement",
    "system_tweak",
]

EffectType = Literal["effective", "ineffective", "unclear"]
RiskType = Literal["safe", "risky"]

SymptomType = Literal[
    "crash_on_launch",
    "black_screen",
    "stutter",
    "no_audio",
    "controller_issue",
    "launcher_crash",
    "anti_cheat_fail",
    "other",
]

ConditionKind = Literal[
    "gpu_vendor",
    "symptom",
    "display_server",
    "distro",
    "proton_version",
]


class Condition(BaseModel):
    kind: ConditionKind
    value: str


class Action(BaseModel):
    type: ActionType
    value: str
    detail: str | None = None
    reported_effect: EffectType = "unclear"
    conditions: list[Condition] = Field(default_factory=list)
    risk: RiskType = "safe"


class Observation(BaseModel):
    symptom: SymptomType = "other"
    description: str = ""
    hardware_specific: bool = False


class ExtractionResult(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    useful: bool = False
