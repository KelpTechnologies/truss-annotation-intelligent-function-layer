"""
Validation Engine
=================

Core validation engine that runs rules against classification responses.
"""

import logging
from typing import Dict, Any, List, Optional, Set

from .models import ValidationInfo, ValidationError
from .rules import create_rule, BaseRule

logger = logging.getLogger(__name__)


class Validator:
    """Validation engine for classification responses."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration dict with 'rules', 'logic', 'output_types', etc.
        """
        self.config = config
        self.rules: List[BaseRule] = []
        self.logic = config.get('logic', 'AND').upper()
        self.output_types = config.get('output_types', {})
        
        rules_config = config.get('rules', [])
        for rule_config in rules_config:
            if rule_config.get('enabled', True):
                try:
                    rule = create_rule(rule_config)
                    self.rules.append(rule)
                except Exception as e:
                    logger.error(f"Failed to create rule {rule_config.get('id', 'unknown')}: {e}")
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any], raw_output: Optional[str] = None) -> ValidationInfo:
        """
        Validate data against configured rules.
        
        Args:
            data: Parsed response data (dict)
            context: Validation context (e.g., valid_ids, property_type)
            raw_output: Raw LLM response string (for debugging)
            
        Returns:
            ValidationInfo with validation results
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        if self.output_types:
            output_type_valid = False
            output_type_errors = []
            output_type_warnings = []
            
            for output_type_name, output_type_config in self.output_types.items():
                type_rules = output_type_config.get('rules', [])
                type_logic = output_type_config.get('logic', 'AND').upper()
                
                type_rule_instances = [r for r in self.rules if r.rule_id in type_rules]
                
                if not type_rule_instances:
                    continue
                
                type_errors = []
                type_warnings = []
                
                for rule in type_rule_instances:
                    error = rule.validate(data, context)
                    if error:
                        if error.severity == 'error':
                            type_errors.append(error)
                        else:
                            type_warnings.append(error)
                
                if type_logic == 'AND':
                    type_passes = len(type_errors) == 0
                else:
                    type_passes = len(type_errors) < len(type_rule_instances)
                
                if type_passes:
                    output_type_valid = True
                    warnings.extend(type_warnings)
                    break
                else:
                    output_type_errors.extend(type_errors)
                    output_type_warnings.extend(type_warnings)
            
            if output_type_valid:
                errors = []
            else:
                errors = output_type_errors[:len(output_type_errors) // len(self.output_types) + 1] if self.output_types else []
                warnings = output_type_warnings
            
            is_valid = output_type_valid
            category = "success" if is_valid else "output_type_validation_failed"
        
        else:
            for rule in self.rules:
                error = rule.validate(data, context)
                if error:
                    if error.severity == 'error':
                        errors.append(error)
                    else:
                        warnings.append(error)
            
            if self.logic == 'AND':
                is_valid = len(errors) == 0
            else:
                is_valid = len(errors) < len(self.rules)
            
            if is_valid:
                category = "success"
            elif any(e.rule_id == 'taxonomy' or 'taxonomy' in e.rule_id.lower() for e in errors):
                category = "hallucinated_id"
            elif any(e.rule_id == 'json_format' or 'json' in e.rule_id.lower() for e in errors):
                category = "parsing_failed"
            else:
                category = "validation_failed"
        
        return ValidationInfo(
            is_valid=is_valid,
            category=category,
            errors=errors,
            warnings=warnings,
            raw_output=raw_output
        )
