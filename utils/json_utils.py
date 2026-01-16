"""
JSON utilities for schema validation and repair.

Provides tools to:
- Validate JSON against schemas
- Repair common JSON formatting errors
- Extract JSON from mixed text
"""

import json
import re
from typing import Any, Dict, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator


def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown or other content.

    Looks for JSON between ```json and ``` blocks, or standalone JSON objects.

    Args:
        text: Input text possibly containing JSON

    Returns:
        Extracted JSON string or None
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0]

    # Try to find standalone JSON object
    json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        # Return the longest match (likely most complete)
        return max(matches, key=len)

    # Try to find JSON array
    array_pattern = r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])"
    matches = re.findall(array_pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len)

    return None


def repair_json(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Attempt to repair common JSON formatting errors.

    Common fixes:
    - Remove trailing commas
    - Fix unquoted keys
    - Replace single quotes with double quotes
    - Remove comments

    Args:
        text: Potentially malformed JSON string

    Returns:
        Tuple of (success: bool, repaired_json: str | None, error: str | None)
    """
    original_text = text
    errors = []

    try:
        # First, try parsing as-is
        json.loads(text)
        return True, text, None
    except json.JSONDecodeError:
        pass

    # Remove comments (// and /* */)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single quotes with double quotes (simple cases)
    text = text.replace("'", '"')

    # Try to fix unquoted keys (basic heuristic)
    text = re.sub(r"(\w+):", r'"\1":', text)

    # Remove potential markdown artifacts
    text = text.strip().strip("`")

    try:
        json.loads(text)
        return True, text, None
    except json.JSONDecodeError as e:
        return False, None, f"Repair failed: {str(e)}"


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data against a JSON schema.

    Args:
        data: Parsed JSON data (dict or list)
        schema: JSON schema definition

    Returns:
        Tuple of (valid: bool, error_message: str | None)
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, f"Schema validation failed: {e.message}"


def safe_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Safely parse JSON with automatic extraction and repair.

    Combines extract_json() and repair_json() for robust parsing.

    Args:
        text: Text containing or being JSON

    Returns:
        Tuple of (success: bool, data: Any | None, error: str | None)
    """
    # First, try direct parse
    try:
        data = json.loads(text)
        return True, data, None
    except json.JSONDecodeError:
        pass

    # Try extraction
    extracted = extract_json(text)
    if extracted:
        try:
            data = json.loads(extracted)
            return True, data, None
        except json.JSONDecodeError:
            pass

        # Try repair on extracted
        success, repaired, error = repair_json(extracted)
        if success and repaired:
            try:
                data = json.loads(repaired)
                return True, data, None
            except json.JSONDecodeError as e:
                return False, None, f"Parse failed after repair: {str(e)}"

    # Try repair on original
    success, repaired, error = repair_json(text)
    if success and repaired:
        try:
            data = json.loads(repaired)
            return True, data, None
        except json.JSONDecodeError as e:
            return False, None, f"Parse failed: {str(e)}"

    return False, None, "Unable to extract or parse JSON from text"


def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """
    Format a JSON schema nicely for inclusion in prompts.

    Args:
        schema: JSON schema dict

    Returns:
        Formatted string representation
    """
    return json.dumps(schema, indent=2)


def create_simple_schema(
    properties: Dict[str, str], required: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    Create a simple JSON schema from property types.

    Args:
        properties: Dict mapping field names to types (string, number, boolean, etc.)
        required: Optional list of required field names

    Returns:
        JSON schema dict

    Example:
        >>> schema = create_simple_schema(
        ...     {"name": "string", "age": "number", "active": "boolean"},
        ...     required=["name"]
        ... )
    """
    schema = {
        "type": "object",
        "properties": {
            name: {"type": prop_type} for name, prop_type in properties.items()
        },
    }

    if required:
        schema["required"] = required

    return schema


def pydantic_to_json_schema(pydantic_model) -> dict:
    """
    Convert a Pydantic model to JSON schema.

    Args:
        pydantic_model: A Pydantic BaseModel class (not instance)

    Returns:
        JSON schema dict

    Example:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float
        >>> schema = pydantic_to_json_schema(Product)
    """
    return pydantic_model.model_json_schema()


def parse_json_with_pydantic(json_str: str, pydantic_model):
    """
    Parse and validate JSON string using a Pydantic model.

    Args:
        json_str: JSON string to parse
        pydantic_model: Pydantic BaseModel class

    Returns:
        Tuple of (success: bool, data: BaseModel or None, error: str or None)

    Example:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float
        >>> success, product, err = parse_json_with_pydantic('{"name": "X", "price": 9.99}', Product)
    """
    try:
        # Parse JSON first
        success, data, error = safe_parse_json(json_str)
        if not success:
            return False, None, error

        # Validate with Pydantic
        model_instance = pydantic_model.model_validate(data)
        return True, model_instance, None
    except Exception as e:
        return False, None, str(e)


def format_pydantic_schema_for_prompt(pydantic_model) -> str:
    """
    Format a Pydantic model's schema for inclusion in prompts.

    Args:
        pydantic_model: Pydantic BaseModel class

    Returns:
        Formatted schema string

    Example:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float
        >>> schema_str = format_pydantic_schema_for_prompt(Product)
    """
    schema = pydantic_to_json_schema(pydantic_model)
    return format_schema_for_prompt(schema)
