"""
Pydantic Model Registry
=======================

Registry for Pydantic models that can be referenced by name in validation configs.
"""

from typing import Dict, Type, Any
from pydantic import BaseModel

# Registry mapping model names to Pydantic model classes
PYDANTIC_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_pydantic_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a Pydantic model for use in validation configs.
    
    Args:
        name: Name to register the model under
        model_class: Pydantic model class
    """
    PYDANTIC_REGISTRY[name] = model_class


def get_pydantic_model(name: str) -> Type[BaseModel]:
    """
    Retrieve a registered Pydantic model by name.
    
    Args:
        name: The registered name
        
    Returns:
        The Pydantic model class
        
    Raises:
        KeyError: If model not registered
    """
    if name not in PYDANTIC_REGISTRY:
        raise KeyError(f"Pydantic model '{name}' not registered. "
                      f"Available: {list(PYDANTIC_REGISTRY.keys())}")
    return PYDANTIC_REGISTRY[name]


def list_registered_models() -> list:
    """List all registered model names."""
    return list(PYDANTIC_REGISTRY.keys())
