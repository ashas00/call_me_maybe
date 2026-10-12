"""Pydantic models for the input files and the result records"""
import json
from typing import Any
from pydantic import BaseModel, Field

SUPPORTED_TYPES: tuple[str, ...] = ("number", "integer", "string", "boolean")
PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "boolean": (bool,),
}


class ParamType(BaseModel):
    """Type of one function parameter, with optional allowed values"""
    type: str
    enum: list[str | bool | int | float] | None = None

    def is_supported(self) -> bool:
        """Return True if the decoder can generate this parameter"""
        if self.type not in SUPPORTED_TYPES:
            return False
        if self.enum is None:
            return True
        return bool(self.enum) and all(
            isinstance(value, PYTHON_TYPES[self.type])
            and (self.type == "boolean" or not isinstance(value, bool))
            for value in self.enum
        )

    def choices(self) -> list[str] | None:
        """Return the allowed values as JSON text, or None if unrestricted"""
        if self.enum is None:
            return None
        if self.type == "number":
            return [json.dumps(float(value)) for value in self.enum]
        return [json.dumps(value) for value in self.enum]


class ReturnType(BaseModel):
    """Return type metadata for a function definition"""
    type: str


class FunctionDef(BaseModel):
    """One function entry as written in the JSON function definition file"""
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParamType]
    returns: ReturnType | None = None

    def unsupported_parameters(self) -> list[str]:
        """Return the names of parameters the decoder cannot generate"""
        return [
            name
            for name, parameter in self.parameters.items()
            if not parameter.is_supported()
        ]


class PromptItem(BaseModel):
    """One natural-language request from the prompt input file"""
    prompt: str


class FunctionCall(BaseModel):
    """One output record written to the JSON results file"""
    prompt: str
    name: str
    parameters: dict[str, Any]
