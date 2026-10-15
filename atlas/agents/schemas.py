"""Structured-output schemas.

CONCEPT: structured outputs turn the model into a typed function. Passing a
Pydantic model to `messages.parse()` constrains generation to the schema, so
the plan and the critique arrive as validated Python objects that ordinary
code can branch on - no regex parsing, no "please answer in JSON".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Specialist = Literal["researcher", "analyst", "librarian"]


class Subtask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1'.")
    title: str
    specialist: Specialist
    instructions: str = Field(description="Self-contained instructions for the specialist.")
    depends_on: list[str] = Field(default_factory=list, description="Ids of subtasks whose output is needed.")


class ResearchPlan(BaseModel):
    objective: str
    subtasks: list[Subtask]
    success_criteria: list[str]


class Critique(BaseModel):
    score: int = Field(description="Overall quality 1-10.")
    passes: bool
    strengths: list[str]
    issues: list[str]
    revision_instructions: str


class JudgeVerdict(BaseModel):
    """Used by the eval harness (LLM-as-judge)."""

    score: int = Field(description="1-5 against the rubric.")
    reasoning: str
    passed: bool
