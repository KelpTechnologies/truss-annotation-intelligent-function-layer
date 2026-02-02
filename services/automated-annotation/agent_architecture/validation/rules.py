"""
Validation Rules
================

Individual rule implementations for validating classification responses.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Set, List
from abc import ABC, abstractmethod

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from .models import ValidationError
from .pydantic_registry import get_pydantic_model

logger = logging.getLogger(__name__)


class BaseRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, rule_id: str, config: Dict[str, Any]):
        self.rule_id = rule_id
        self.config = config
        self.severity = config.get('severity', 'error')
        self.message_template = config.get('message', 'Validation failed')
        self.enabled = config.get('enabled', True)
        self.params = config.get('params', {})
    
    @abstractmethod
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        pass
    
    def _format_message(self, **kwargs) -> str:
        try:
            return self.message_template.format(**kwargs)
        except (KeyError, ValueError):
            return self.message_template


class JsonFormatRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        if not isinstance(data, dict):
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(),
                severity=self.severity,
                details={'expected': 'dict', 'actual': type(data).__name__}
            )
        return None


class RequiredFieldsRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        required_fields = self.params.get('fields', [])
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(fields=', '.join(missing_fields)),
                severity=self.severity,
                details={'missing_fields': missing_fields, 'required_fields': required_fields}
            )
        return None


class FieldTypeRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field_types = self.params.get('field_types', {})
        type_errors = []
        
        for field, expected_type in field_types.items():
            if field not in data:
                continue
            
            value = data[field]
            expected_type_name = expected_type.lower()
            
            if expected_type_name == 'number' and not isinstance(value, (int, float)):
                type_errors.append(f"{field}: expected number, got {type(value).__name__}")
            elif expected_type_name == 'integer' and not isinstance(value, int):
                type_errors.append(f"{field}: expected integer, got {type(value).__name__}")
            elif expected_type_name == 'float' and not isinstance(value, (int, float)):
                type_errors.append(f"{field}: expected float, got {type(value).__name__}")
            elif expected_type_name == 'string' and not isinstance(value, str):
                type_errors.append(f"{field}: expected string, got {type(value).__name__}")
            elif expected_type_name == 'list' and not isinstance(value, list):
                type_errors.append(f"{field}: expected list, got {type(value).__name__}")
            elif expected_type_name == 'dict' and not isinstance(value, dict):
                type_errors.append(f"{field}: expected dict, got {type(value).__name__}")
        
        if type_errors:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(errors='; '.join(type_errors)),
                severity=self.severity,
                details={'type_errors': type_errors}
            )
        return None


class RangeRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field = self.params.get('field')
        min_val = self.params.get('min')
        max_val = self.params.get('max')
        
        if not field or field not in data:
            return None
        
        value = data[field]
        if not isinstance(value, (int, float)):
            return None
        
        if min_val is not None and value < min_val:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, value=value, min=min_val),
                severity=self.severity,
                field_name=field,
                details={'value': value, 'min': min_val, 'max': max_val}
            )
        
        if max_val is not None and value > max_val:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, value=value, max=max_val),
                severity=self.severity,
                field_name=field,
                details={'value': value, 'min': min_val, 'max': max_val}
            )
        
        return None


class ThresholdRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field = self.params.get('field')
        threshold = self.params.get('threshold')
        comparison = self.params.get('comparison', '>=')
        
        if not field or field not in data:
            return None
        
        value = data[field]
        if not isinstance(value, (int, float)):
            return None
        
        failed = False
        if comparison == '>=' and value < threshold:
            failed = True
        elif comparison == '>' and value <= threshold:
            failed = True
        elif comparison == '<=' and value > threshold:
            failed = True
        elif comparison == '<' and value >= threshold:
            failed = True
        elif comparison == '==' and value != threshold:
            failed = True
        
        if failed:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, value=value, threshold=threshold, comparison=comparison),
                severity=self.severity,
                field_name=field,
                details={'value': value, 'threshold': threshold, 'comparison': comparison}
            )
        
        return None


class TaxonomyRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field = self.params.get('field', 'prediction_id')
        id_field = self.params.get('id_field', field)
        
        if field not in data:
            return None
        
        valid_ids: Set[int] = context.get('valid_ids', set())
        id_value = data[field]
        
        if id_field != field and '.' in id_field:
            parts = id_field.split('.')
            id_value = data
            for part in parts:
                if isinstance(id_value, dict):
                    id_value = id_value.get(part)
                elif isinstance(id_value, list) and part.isdigit():
                    id_value = id_value[int(part)] if int(part) < len(id_value) else None
                else:
                    id_value = None
                if id_value is None:
                    break
        
        if id_value is None:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, id_field=id_field),
                severity=self.severity,
                field_name=field,
                details={'id_field': id_field, 'valid_ids': sorted(list(valid_ids))}
            )
        
        try:
            id_int = int(id_value)
        except (ValueError, TypeError):
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, value=id_value),
                severity=self.severity,
                field_name=field,
                details={'value': id_value, 'expected_type': 'int'}
            )

        if id_int < 0:
            return ValidationError(
                rule_id=self.rule_id,
                message=f"ID {id_int} is invalid (negative IDs not allowed)",
                severity=self.severity,
                field_name=field,
                details={'id': id_int, 'reason': 'negative_id', 'valid_ids': sorted(list(valid_ids))}
            )

        if id_int != 0 and id_int not in valid_ids:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, id=id_int, valid_ids=sorted(list(valid_ids))),
                severity=self.severity,
                field_name=field,
                details={'id': id_int, 'valid_ids': sorted(list(valid_ids))}
            )
        
        return None


class PatternRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field = self.params.get('field')
        pattern = self.params.get('pattern')
        
        if not field or field not in data:
            return None
        
        value = str(data[field])
        
        try:
            if not re.match(pattern, value):
                return ValidationError(
                    rule_id=self.rule_id,
                    message=self._format_message(field=field, value=value, pattern=pattern),
                    severity=self.severity,
                    field_name=field,
                    details={'value': value, 'pattern': pattern}
                )
        except re.error as e:
            logger.warning(f"Invalid regex pattern in PatternRule {self.rule_id}: {e}")
            return None
        
        return None


class EnumRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        field = self.params.get('field')
        values = self.params.get('values', [])
        
        if not field or field not in data:
            return None
        
        value = data[field]
        
        if value not in values:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(field=field, value=value, allowed_values=values),
                severity=self.severity,
                field_name=field,
                details={'value': value, 'allowed_values': values}
            )
        
        return None


class ToolCallFormatRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        required_keys = self.params.get('required_keys', ['name', 'arguments'])
        
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(missing_keys=', '.join(missing_keys)),
                severity=self.severity,
                details={'missing_keys': missing_keys, 'required_keys': required_keys}
            )
        
        return None


class JsonSchemaRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        if not HAS_JSONSCHEMA:
            logger.warning(f"jsonschema library not installed, skipping JsonSchemaRule {self.rule_id}")
            return None
        
        schema = self.params.get('schema')
        if not schema:
            logger.warning(f"JsonSchemaRule {self.rule_id} missing 'schema' param")
            return None
        
        try:
            jsonschema.validate(data, schema)
            return None
        except jsonschema.ValidationError as e:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(error=str(e.message), path='.'.join(str(p) for p in e.path)),
                severity=self.severity,
                details={'schema_error': str(e.message), 'path': list(e.path)}
            )
        except Exception as e:
            logger.error(f"JsonSchemaRule {self.rule_id} failed: {e}")
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(error=str(e)),
                severity=self.severity,
                details={'error': str(e)}
            )


class PydanticClassRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        class_name = self.params.get('class_name')
        if not class_name:
            logger.warning(f"PydanticClassRule {self.rule_id} missing 'class_name' param")
            return None
        
        try:
            model_class = get_pydantic_model(class_name)
            model_class(**data)
            return None
        except ValueError as e:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(class_name=class_name, error=str(e)),
                severity=self.severity,
                details={'class_name': class_name, 'error': str(e)}
            )
        except Exception as e:
            logger.error(f"PydanticClassRule {self.rule_id} failed: {e}")
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(class_name=class_name, error=str(e)),
                severity=self.severity,
                details={'class_name': class_name, 'error': str(e)}
            )


class RequiredKeysRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        keys = self.params.get('keys', [])
        if not keys:
            return None
        
        missing_keys = [key for key in keys if key not in data]
        
        if missing_keys:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(missing_keys=', '.join(missing_keys)),
                severity=self.severity,
                details={'missing_keys': missing_keys, 'required_keys': keys}
            )
        
        return None


class KeyValueTypeRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        keys = self.params.get('keys', [])
        expected_type = self.params.get('expected_type', '').lower()
        
        if not keys or not expected_type:
            return None
        
        type_errors = []
        type_map = {
            'array': list, 'list': list, 'string': str, 'str': str,
            'number': (int, float), 'int': int, 'integer': int,
            'float': (int, float), 'object': dict, 'dict': dict,
            'boolean': bool, 'bool': bool
        }
        
        expected_type_class = type_map.get(expected_type)
        if not expected_type_class:
            logger.warning(f"KeyValueTypeRule {self.rule_id} unknown expected_type: {expected_type}")
            return None
        
        for key in keys:
            if key not in data:
                continue
            
            value = data[key]
            actual_type = type(value).__name__
            
            if isinstance(expected_type_class, tuple):
                if not isinstance(value, expected_type_class):
                    type_errors.append(f"{key}: expected {expected_type}, got {actual_type}")
            elif not isinstance(value, expected_type_class):
                type_errors.append(f"{key}: expected {expected_type}, got {actual_type}")
        
        if type_errors:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(errors='; '.join(type_errors)),
                severity=self.severity,
                details={'type_errors': type_errors, 'expected_type': expected_type}
            )
        
        return None


class ValuesInContextRule(BaseRule):
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ValidationError]:
        context_key = self.params.get('context_key')
        flatten_arrays = self.params.get('flatten_arrays', False)
        field = self.params.get('field')
        
        if not context_key:
            logger.warning(f"ValuesInContextRule {self.rule_id} missing 'context_key' param")
            return None
        
        valid_values = context.get(context_key, [])
        if not isinstance(valid_values, (list, set)):
            logger.warning(f"ValuesInContextRule {self.rule_id} context_key '{context_key}' is not a list/set")
            return None
        
        valid_set = set(valid_values) if isinstance(valid_values, list) else valid_values
        invalid_values = []
        
        if field:
            if field not in data:
                return None
            
            value = data[field]
            if flatten_arrays and isinstance(value, list):
                for item in value:
                    if item not in valid_set:
                        invalid_values.append(item)
            else:
                if value not in valid_set:
                    invalid_values.append(value)
        else:
            if flatten_arrays:
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if item not in valid_set:
                                invalid_values.append(item)
                    elif value not in valid_set:
                        invalid_values.append(value)
            else:
                for key, value in data.items():
                    if value not in valid_set:
                        invalid_values.append(value)
        
        if invalid_values:
            return ValidationError(
                rule_id=self.rule_id,
                message=self._format_message(
                    value=invalid_values[0] if len(invalid_values) == 1 else f"{len(invalid_values)} values",
                    context=list(valid_set)[:10]
                ),
                severity=self.severity,
                details={'invalid_values': invalid_values[:10], 'valid_values': list(valid_set)[:20]}
            )
        
        return None


RULE_REGISTRY = {
    'json_format': JsonFormatRule,
    'required_fields': RequiredFieldsRule,
    'required_keys': RequiredKeysRule,
    'field_types': FieldTypeRule,
    'key_value_type': KeyValueTypeRule,
    'range': RangeRule,
    'threshold': ThresholdRule,
    'taxonomy': TaxonomyRule,
    'pattern': PatternRule,
    'enum': EnumRule,
    'tool_call_format': ToolCallFormatRule,
    'json_schema': JsonSchemaRule,
    'pydantic_class': PydanticClassRule,
    'values_in_context': ValuesInContextRule,
}


def create_rule(rule_config: Dict[str, Any]) -> BaseRule:
    rule_type = rule_config.get('type')
    rule_id = rule_config.get('id', f"rule_{rule_type}")
    
    if rule_type not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule type: {rule_type}")
    
    rule_class = RULE_REGISTRY[rule_type]
    return rule_class(rule_id=rule_id, config=rule_config)
