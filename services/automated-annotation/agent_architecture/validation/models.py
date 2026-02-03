"""
Validation Models
=================

Data structures for validation results, agent status, and error reporting.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..base_agent import ClassificationResponse


class AgentStatus(Enum):
    """Status of agent classification attempt."""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    PARSING_FAILED = "parsing_failed"
    LLM_ERROR = "llm_error"
    INVALID_INPUT = "invalid_input"


@dataclass
class ValidationError:
    """Individual validation error or warning."""
    rule_id: str
    message: str
    field_name: Optional[str] = None  # Renamed from 'field' to avoid conflict with dataclasses.field
    severity: str = "error"  # "error" or "warning"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationInfo:
    """Validation results for a classification response."""
    is_valid: bool
    category: str  # "success", "hallucinated_id", "parsing_failed", etc.
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    raw_output: Optional[str] = None  # Raw LLM response for debugging


@dataclass
class ErrorReport:
    """Detailed error information for failed classifications."""
    error_type: str  # Category of failure
    message: str  # Human-readable message
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False  # Hint to orchestrator if retry might help


@dataclass
class AgentResult:
    """Unified return type for all agent classification calls."""
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None  # Parsed LLM output as dict (None if failed)
    validation_info: Optional[ValidationInfo] = None  # Validation details (always populated if validation ran)
    error_report: Optional[ErrorReport] = None  # Detailed error info (None if success)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Processing time, attempt count, etc.
    schema: Optional[Dict[str, Any]] = None  # Schema content for ID->name mapping in orchestration
