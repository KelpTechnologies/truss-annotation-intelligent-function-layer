"""
Measurement Conversion and Matching Utilities
=============================================

Utilities for converting measurements to cm and finding closest matches.
"""

from typing import Dict, Any, Optional, List
import math


def convert_to_cm(value: float, unit: str) -> float:
    """
    Convert measurement value to centimeters.
    
    Args:
        value: Measurement value
        unit: Unit ('inches', 'cm', 'mm', 'in')
        
    Returns:
        Value in centimeters
    """
    unit_lower = unit.lower().strip()
    
    if unit_lower in ['cm', 'centimeter', 'centimetre']:
        return value
    elif unit_lower in ['mm', 'millimeter', 'millimetre']:
        return value / 10.0
    elif unit_lower in ['inches', 'inch', 'in', '"']:
        return value * 2.54
    else:
        raise ValueError(f"Unknown unit: {unit}. Supported: inches, cm, mm")


def calculate_euclidean_distance(
    measurements1: Dict[str, float],
    measurements2: Dict[str, float]
) -> float:
    """
    Calculate Euclidean distance between two sets of measurements.
    
    Args:
        measurements1: Dict with height, width, length (in cm)
        measurements2: Dict with height, width, length (in cm)
        
    Returns:
        Euclidean distance
    """
    dimensions = ['height', 'width', 'length']
    
    # Handle None values by treating as 0
    sum_squared_diff = 0.0
    for dim in dimensions:
        val1 = measurements1.get(dim) or 0.0
        val2 = measurements2.get(dim) or 0.0
        diff = val1 - val2
        sum_squared_diff += diff * diff
    
    return math.sqrt(sum_squared_diff)


def find_closest_size_match(
    extracted_measurements: Dict[str, float],
    size_options: List[Dict[str, Any]],
    unit: str = 'cm'
) -> Optional[Dict[str, Any]]:
    """
    Find the closest size match using Euclidean distance.
    
    Args:
        extracted_measurements: Dict with height, width, length (in specified unit)
        size_options: List of size options from BigQuery with id, size, height, width, length
        unit: Unit of extracted_measurements ('inches', 'cm', 'mm')
        
    Returns:
        Closest match dict with: id, size, distance, or None if no valid options
    """
    if not size_options:
        return None
    
    # Convert extracted measurements to cm
    measurements_cm = {}
    for dim in ['height', 'width', 'length']:
        if dim in extracted_measurements and extracted_measurements[dim] is not None:
            measurements_cm[dim] = convert_to_cm(extracted_measurements[dim], unit)
        else:
            measurements_cm[dim] = None
    
    # Calculate distance to each option
    best_match = None
    best_distance = float('inf')
    
    for option in size_options:
        option_measurements = {
            'height': option.get('height'),
            'width': option.get('width'),
            'length': option.get('length')
        }
        
        # Skip if option has no measurements
        if not any(option_measurements.values()):
            continue
        
        # Calculate distance (handle None values)
        distance = calculate_euclidean_distance(measurements_cm, option_measurements)
        
        if distance < best_distance:
            best_distance = distance
            best_match = {
                'id': option['id'],
                'size': option['size'],
                'distance': distance,
                'option_measurements': option_measurements,
                'extracted_measurements_cm': measurements_cm
            }
    
    return best_match
