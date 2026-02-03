"""
LLM Annotation System - Base Agent (API-only mode)
==================================================

A generalized agent for fashion product metadata annotation using Google GenAI.
Supports structured outputs, prompt templating, and retry logic with Gemini models.
Modified for API-only operation - no local file dependencies.

Uses google-genai SDK directly instead of LangChain for minimal dependencies (~5MB vs ~230MB).
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import base64
import requests
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import pandas as pd

from .validation import (
    AgentResult,
    AgentStatus,
    ValidationInfo,
    ErrorReport,
    Validator,
    register_pydantic_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionScore(BaseModel):
    """Individual prediction with raw score."""
    id: int = Field(description="Taxonomy ID (integer)")
    score: float = Field(description="Raw model score (0-1)", ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Structured response for all classification tasks with top-k scoring."""
    prediction_id: int = Field(description="Primary prediction taxonomy ID (integer)")
    scores: List[PredictionScore] = Field(
        description="Top-k predictions with raw scores",
        min_items=1,
        max_items=5
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation for the primary prediction (max 30 words)"
    )

    @property
    def primary(self) -> str:
        """Return primary prediction as formatted string."""
        return f"ID {self.prediction_id}"

    @property
    def alternatives(self) -> List[str]:
        """Return alternative predictions as formatted strings."""
        return [f"ID {score.id}" for score in self.scores[1:]]  # Exclude primary

    @property
    def confidence(self) -> float:
        """Return confidence score for primary prediction."""
        primary_score = next((s.score for s in self.scores if s.id == self.prediction_id), 0.0)
        return round(primary_score, 3)


# Register Pydantic models for validation
register_pydantic_model("ClassificationResponse", ClassificationResponse)
register_pydantic_model("PredictionScore", PredictionScore)


class LLMAnnotationAgent:
    """
    LLM-powered agent for fashion product metadata annotation.
    Modified for API-only operation.
    """

    def __init__(
        self,
        full_config: Dict[str, Any],
        log_IO: bool = False
    ):
        """
        Initialize agent with complete configuration bundle.
        
        Args:
            full_config: Complete agent configuration containing:
                - config_id: str
                - model_config: dict (model, project_id, location, temperature, etc.)
                - validation_config: dict (rules, logic)
                - stopping_conditions: dict (stop_on, max_attempts)
                - schema: dict (schema_id, schema_content, schema_metadata) - optional
                - prompt_template: dict (templates, model_overrides) - optional
                - prompt_template_id: str - optional
                - schema_id: str - optional
            log_IO: Whether to log I/O data
        """
        # Store full config for reference
        self.full_config = full_config
        self.log_IO = log_IO
        
        # Extract required components
        self.config_id = full_config.get('config_id', 'unknown')
        self.model_config = full_config.get('model_config', {})
        self.validation_config = full_config.get('validation_config')
        self.stopping_conditions = full_config.get('stopping_conditions', {
            "stop_on": "validated_output",
            "max_attempts": 1
        })
        
        # Validate required configs
        if not self.validation_config:
            raise ValueError("validation_config is required in full_config")
        
        # Extract optional components
        self.prompt_template = full_config.get('prompt_template')
        self.schema = full_config.get('schema')
        
        # Extract model parameters
        self.model_name = self.model_config.get('model', 'gemini-2.5-flash')
        self.project_id = self.model_config.get('project_id', 'truss-data-science')
        self.location = self.model_config.get('location', 'us-central1')
        
        # Initialize validator
        try:
            self.validator = Validator(self.validation_config)
            logger.info("Validation system initialized")
        except Exception as e:
            raise ValueError(f"Failed to initialize validator: {e}") from e
        
        # Extract valid IDs from schema (for taxonomy validation)
        self._valid_ids = set()
        if self.schema and self.schema.get('schema_content'):
            self._valid_ids = self._extract_valid_ids_from_schema(self.schema['schema_content'])
        
        # Build base validation context (can be extended per-call)
        self._base_context = {
            'valid_ids': self._valid_ids,
            'schema_content': self.schema.get('schema_content', {}) if self.schema else {}
        }
        
        # Initialize Google GenAI client (uses VertexAI backend)
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
        # Store generation config parameters
        self.temperature = self.model_config.get('temperature', 0.1)
        self.max_output_tokens = self.model_config.get('max_output_tokens', 2048)
        
        # Set model-specific defaults for better performance
        model_defaults = {
            "gemini-2.5-pro": {"temperature": 0.1, "max_output_tokens": 5000},
            "gemini-2.5-flash": {"temperature": 0.1, "max_output_tokens": 5000},
            "gemini-2.5-flash-lite": {"temperature": 0.1, "max_output_tokens": 5000},
        }
        
        if self.model_name in model_defaults:
            self.temperature = model_defaults[self.model_name].get("temperature", self.temperature)
            self.max_output_tokens = model_defaults[self.model_name].get("max_output_tokens", self.max_output_tokens)
        
        logger.info(f"Initialized agent '{self.config_id}' with model {self.model_name}")

    def _extract_valid_ids_from_schema(self, schema_content: dict) -> set:
        """
        Extract valid taxonomy IDs from schema content.
        
        Args:
            schema_content: Dict mapping ID -> item data
                e.g., {"1": {"name": "Leather"}, "2": {"name": "Canvas"}}
        
        Returns:
            Set of valid integer IDs
        """
        valid_ids = set()
        for key in schema_content.keys():
            try:
                valid_ids.add(int(key))
            except (ValueError, TypeError):
                logger.warning(f"Non-integer schema key: {key}")
        return valid_ids

    def load_prompt_template(self, template_path: str) -> None:
        """Load modular prompt template from JSON file."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_config = json.load(f)

            # Store the full template config for dynamic use
            self.prompt_template = template_config

            # Extract model-specific overrides if they exist
            model_overrides = template_config.get('model_overrides', {})
            self.model_config_overrides = model_overrides.get(self.model_name, {})

        except Exception as e:
            logger.error(f"Failed to load prompt template from {template_path}: {e}")
            raise

    def prepare_multimodal_input(self, image_url: str, text_prompt: str, input_mode: str = "auto") -> List[Any]:
        """
        Prepare multimodal input for Google GenAI based on input mode.

        Args:
            image_url: URL of the image to analyze
            text_prompt: Text prompt to send to the model
            input_mode: 'image-only', 'text-only', 'multimodal', or 'auto'

        Returns:
            List of content parts for Google GenAI
        """
        # Determine effective input mode
        if input_mode == "auto":
            # Auto-detect based on available inputs
            has_image = bool(image_url and image_url.strip())
            has_text = bool(text_prompt and text_prompt.strip())

            if has_image and has_text:
                effective_mode = "multimodal"
            elif has_image:
                effective_mode = "image-only"
            elif has_text:
                effective_mode = "text-only"
            else:
                raise ValueError("No image or text provided for classification")
        else:
            effective_mode = input_mode

        # Build content parts based on mode
        if effective_mode == "image-only":
            if not image_url:
                raise ValueError("Image URL required for image-only mode")

            # Download image as bytes
            image_bytes = self._download_image_bytes(image_url)
            return [
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                text_prompt
            ]

        elif effective_mode == "text-only":
            return [text_prompt]

        elif effective_mode == "multimodal":
            if not image_url:
                raise ValueError("Image URL required for multimodal mode")

            # Download image as bytes
            image_bytes = self._download_image_bytes(image_url)
            return [
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                text_prompt
            ]
        else:
            raise ValueError(f"Unknown input mode: {effective_mode}")

    def _download_image_bytes(self, image_url: str) -> bytes:
        """
        Download image from URL and return raw bytes.

        Args:
            image_url: URL of the image to download

        Returns:
            Raw image bytes (JPEG format)
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Validate it's an image
            img = Image.open(BytesIO(response.content))
            img.verify()  # Validate image integrity

            # Re-open after verify
            img = Image.open(BytesIO(response.content))

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            return buffer.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to download image from {image_url}: {e}")

    def _download_and_encode_image(self, image_url: str) -> str:
        """
        Download image from URL and return base64 encoded string.
        (Legacy method - kept for backward compatibility)

        Args:
            image_url: URL of the image to download

        Returns:
            Base64 encoded image data
        """
        image_bytes = self._download_image_bytes(image_url)
        return base64.b64encode(image_bytes).decode('utf-8')

    def _format_schema_for_prompt(self, schema_content: dict) -> str:
        """Format schema content for inclusion in prompt."""
        lines = []
        for item_id, item_data in schema_content.items():
            name = item_data.get('name', 'Unknown')
            description = item_data.get('description', '')
            if description:
                lines.append(f"ID {item_id}: {name} - {description}")
            else:
                lines.append(f"ID {item_id}: {name}")
        return "\n".join(lines)

    def _get_response_preview(self, response, max_chars: int = 200) -> str:
        """Get preview of LLM response for error reporting."""
        raw = self._get_raw_response_text(response)
        if len(raw) > max_chars:
            return raw[:max_chars] + "..."
        return raw

    def _get_raw_response_text(self, response) -> str:
        """
        Extract raw text from Google GenAI response.
        
        Handles GenerateContentResponse from google-genai SDK.
        """
        # Google GenAI response - use .text property
        if hasattr(response, 'text'):
            return response.text
        
        # Google GenAI response with candidates
        if hasattr(response, 'candidates') and response.candidates:
            try:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Concatenate all text parts
                        texts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                texts.append(part.text)
                        if texts:
                            return ''.join(texts)
            except Exception:
                pass

        # Dict-shaped responses
        if isinstance(response, dict):
            if 'text' in response and isinstance(response['text'], str):
                return response['text']
            if 'content' in response and isinstance(response['content'], str):
                return response['content']

        return str(response)

    def _call_llm_with_image(self, prompt: str, image_url: str):
        """Call LLM with image input using Google GenAI."""
        # Download image as bytes
        image_bytes = self._download_image_bytes(image_url)
        
        # Build multimodal content for Google GenAI
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ]
        
        # Call Google GenAI
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        )

    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute agent task based on input data and configuration.
        
        This is the main entry point for all agent operations. The specific behavior
        (classification, config generation, etc.) is determined by the agent's
        configuration and prompt template.
        
        Args:
            input_data: Input for the task. Common keys:
                - image_url: str (optional) - URL of image to analyze
                - text_input: str (optional) - Text input (product info, raw text, etc.)
                - text_metadata: str (optional) - DEPRECATED: use text_input instead
                - input_text: str (optional) - DEPRECATED: use text_input instead
                - item_id: str (optional) - Item identifier for logging
                - garment_id: str (optional) - DEPRECATED: use item_id instead
                - input_mode: str (optional) - "auto", "image-only", "text-only", "multimodal"
                - format_as_product_info: bool (optional) - If True, prefix text with "Product Information:"
            
            context: Additional context for validation. Merged with base context.
                e.g., {"csv_columns": ["col1", "col2"]} for CSV config generation
        
        Returns:
            AgentResult with:
                - status: AgentStatus
                - result: dict (the parsed LLM output)
                - validation_info: ValidationInfo
                - error_report: ErrorReport (if failed)
                - metadata: dict (attempt count, timing, etc.)
        """
        import time
        start_time = time.time()
        
        # Normalize input field names (backward compatibility)
        # Consolidate text fields: text_input > text_metadata > input_text
        text_content = (
            input_data.get('text_input') or 
            input_data.get('text_metadata') or 
            input_data.get('input_text') or 
            ''
        )
        # Consolidate ID field: item_id > garment_id
        item_id = input_data.get('item_id') or input_data.get('garment_id')
        
        # Validate input based on mode
        input_mode = input_data.get('input_mode', 'auto')
        has_image = bool(input_data.get('image_url'))
        has_text = bool(text_content)
        
        # Validate text-only mode has text
        if input_mode == 'text-only' and not has_text:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error_report=ErrorReport(
                    error_type="missing_text_input",
                    message="text-only mode requires text_input, text_metadata, or input_text to be provided",
                    details={"input_mode": input_mode, "has_text": has_text},
                    recoverable=False
                ),
                metadata={"duration_seconds": time.time() - start_time, "item_id": item_id},
                schema=self.schema
            )
        
        # Validate image-only mode has image
        if input_mode == 'image-only' and not has_image:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error_report=ErrorReport(
                    error_type="missing_image_url",
                    message="image-only mode requires image_url to be provided",
                    details={"input_mode": input_mode, "has_image": has_image},
                    recoverable=False
                ),
                metadata={"duration_seconds": time.time() - start_time, "item_id": item_id},
                schema=self.schema
            )
        
        # Validate multimodal mode has image
        if input_mode == 'multimodal' and not has_image:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error_report=ErrorReport(
                    error_type="missing_image_url",
                    message="multimodal mode requires image_url to be provided",
                    details={"input_mode": input_mode, "has_image": has_image},
                    recoverable=False
                ),
                metadata={"duration_seconds": time.time() - start_time, "item_id": item_id},
                schema=self.schema
            )
        
        # Validate auto mode has at least one input
        if input_mode == 'auto' and not has_image and not has_text:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error_report=ErrorReport(
                    error_type="no_input_provided",
                    message="At least one of image_url or text_input must be provided",
                    details={"input_mode": input_mode, "has_image": has_image, "has_text": has_text},
                    recoverable=False
                ),
                metadata={"duration_seconds": time.time() - start_time, "item_id": item_id},
                schema=self.schema
            )
        
        # Store normalized values back for _build_prompt
        input_data['_text_content'] = text_content
        input_data['_item_id'] = item_id
        
        # 1. Build validation context (base + call-specific)
        validation_context = {**self._base_context}
        if context:
            validation_context.update(context)
        
        # 2. Build prompt
        try:
            prompt = self._build_prompt(input_data)
        except Exception as e:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error_report=ErrorReport(
                    error_type="prompt_build_failed",
                    message=f"Failed to build prompt: {e}",
                    details={"input_data_keys": list(input_data.keys())},
                    recoverable=False
                ),
                metadata={"duration_seconds": time.time() - start_time},
                schema=self.schema
            )
        
        # 3. Call LLM
        try:
            image_url = input_data.get('image_url')
            if image_url:
                response = self._call_llm_with_image(prompt, image_url)
            else:
                # Text-only call using Google GenAI
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    )
                )
        except Exception as e:
            return AgentResult(
                status=AgentStatus.LLM_ERROR,
                error_report=ErrorReport(
                    error_type="llm_error",
                    message=str(e),
                    details={},
                    recoverable=True
                ),
                metadata={"duration_seconds": time.time() - start_time},
                schema=self.schema
            )
        
        # 4. Parse JSON response
        parsed, parse_error = self._parse_json_response(response)
        
        # 4.5. Clean up "Unknown" values - replace with empty string/null
        if parsed is not None:
            parsed = self._strip_unknown_values(parsed)
        
        if parsed is None:
            # Print parsing failure details + raw prompt input
            print(f"\n[PARSING FAILED] {parse_error or 'Failed to parse JSON from response'}")
            print(f"Prompt length: {len(prompt)} chars")
            if len(prompt) <= 4000:
                print("Prompt Sent to Agent:\n" + prompt)
            else:
                print("Prompt Sent to Agent (truncated):\n" + prompt[:4000] + "\n... [truncated] ...")
            print(f"LLM response type: {type(response).__name__}")
            print("Raw LLM Response Preview:\n" + self._get_response_preview(response, max_chars=800))

            return AgentResult(
                status=AgentStatus.PARSING_FAILED,
                error_report=ErrorReport(
                    error_type="parsing_failed",
                    message=parse_error or "Failed to parse JSON from response",
                    details={"raw_response": self._get_response_preview(response)},
                    recoverable=True
                ),
                metadata={"duration_seconds": time.time() - start_time},
                schema=self.schema
            )
        
        # 5. Validate response
        raw_response_text = self._get_response_preview(response, max_chars=1000)
        validation_info = self.validator.validate(parsed, validation_context, raw_response_text)
        
        # 6. Build result
        duration = time.time() - start_time
        
        if validation_info.is_valid:
            return AgentResult(
                status=AgentStatus.SUCCESS,
                result=parsed,
                validation_info=validation_info,
                error_report=None,
                metadata={
                    "attempt": 1,
                    "duration_seconds": round(duration, 3),
                    "item_id": input_data.get('_item_id')
                },
                schema=self.schema
            )
        else:
            # Get raw response for error reporting
            raw_response_preview = self._get_response_preview(response, max_chars=500)
            
            return AgentResult(
                status=AgentStatus.VALIDATION_FAILED,
                result=parsed,  # Include parsed result even on validation failure
                validation_info=validation_info,
                error_report=ErrorReport(
                    error_type="validation_failed",
                    message=f"Validation failed: {validation_info.category}. Raw response: {raw_response_preview}",
                    details={
                        "errors": [{"rule_id": e.rule_id, "message": e.message} for e in validation_info.errors],
                        "raw_response": raw_response_preview,
                        "parsed_output": parsed
                    },
                    recoverable=True
                ),
                metadata={
                    "attempt": 1,
                    "duration_seconds": round(duration, 3),
                    "item_id": input_data.get('_item_id')
                },
                schema=self.schema
            )

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Build prompt based on input data and prompt template.
        
        Handles different input scenarios:
        - Image + text (multimodal classification)
        - Image only (image classification)
        - Text only (text analysis, config generation)
        
        Args:
            input_data: Dict containing input fields
        
        Returns:
            Formatted prompt string
        """
        # Get normalized text content (set by execute())
        text_content = input_data.get('_text_content', '')
        has_image = bool(input_data.get('image_url'))
        has_text = bool(text_content)
        
        # Determine whether to format as product info
        format_as_product_info = input_data.get('format_as_product_info')
        if format_as_product_info is None:
            format_as_product_info = bool(input_data.get('text_metadata') or input_data.get('text_input'))
        
        input_mode = input_data.get('input_mode', 'auto')
        if input_mode == 'auto':
            if has_image and has_text:
                input_mode = 'multimodal'
            elif has_image:
                input_mode = 'image-only'
            else:
                input_mode = 'text-only'
        
        # Build prompt from template
        template = None
        if self.prompt_template:
            if any(key in self.prompt_template for key in ['context_intro', 'instructions', 'system_message', 'output_format']):
                template = self.prompt_template
            elif 'template_content' in self.prompt_template:
                template = self.prompt_template['template_content']
            elif 'templates' in self.prompt_template:
                template = self.prompt_template['templates'].get('simple', 
                           self.prompt_template['templates'].get('standard', {}))
        
        if template:
            prompt_parts = []
            
            if 'system_message' in template:
                prompt_parts.append(template['system_message'])
            
            if 'context_intro' in template:
                prompt_parts.append(template['context_intro'])
            
            if self.schema and self.schema.get('schema_content'):
                context_str = self._format_schema_for_prompt(self.schema['schema_content'])
                prompt_parts.append(f"Available Options:\n{context_str}")
            
            if has_text and input_mode in ['multimodal', 'text-only']:
                if format_as_product_info:
                    prompt_parts.append(f"Product Information:\n{text_content}")
                else:
                    prompt_parts.append(text_content)
            
            if 'instructions' in template:
                prompt_parts.append(template['instructions'])
            
            if 'output_format' in template:
                prompt_parts.append(f"Output Format:\n{template['output_format']}")
            
            return "\n\n".join(prompt_parts)
        
        # Fallback prompt (no template)
        fallback_parts = ["Analyze the provided input and return a JSON response."]
        
        if has_text:
            if format_as_product_info:
                fallback_parts.append(f"Product Information:\n{text_content}")
            else:
                fallback_parts.append(f"Input:\n{text_content}")
        
        fallback_parts.append("Return valid JSON only.")
        
        return "\n\n".join(fallback_parts)

    def classify(
        self,
        image_url: str = None,
        text_metadata: str = None,
        garment_id: str = None,
        root_type_id: int = None,
        brand: str = None,
        input_mode: str = "auto",
        context_data: Dict[str, Any] = None
    ) -> AgentResult:
        """
        DEPRECATED: Use execute() instead.
        Backward-compatible wrapper for classification tasks.
        """
        import warnings
        warnings.warn("classify() is deprecated. Use execute() instead.", DeprecationWarning)
        
        input_data = {
            "image_url": image_url,
            "text_metadata": text_metadata,
            "garment_id": garment_id,
            "input_mode": input_mode
        }
        
        context = None
        if context_data:
            valid_ids = self._extract_valid_ids(context_data)
            context = {"valid_ids": valid_ids}
        
        return self.execute(input_data=input_data, context=context)

    def _extract_valid_ids(self, context_data: dict) -> set:
        """Extract valid taxonomy IDs from context data."""
        valid_ids = set()

        if isinstance(context_data, dict) and 'data' in context_data:
            context_data = context_data['data']

        if isinstance(context_data, dict) and 'context_content' in context_data:
            content = context_data['context_content']
            for item_id, item_data in content.items():
                try:
                    valid_ids.add(int(item_id))
                except (ValueError, TypeError):
                    continue
        else:
            items = context_data.get("materials", context_data.get("items", []))

            for item in items:
                if isinstance(item, str) and item.startswith("ID "):
                    try:
                        id_part = item.split(":")[0].strip()
                        id_num = int(id_part.split()[1])
                        valid_ids.add(id_num)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not extract ID from item '{item}': {e}")

        return valid_ids
    
    def _extract_valid_ids_from_context(self, context_data: dict) -> set:
        """Legacy method name - delegates to _extract_valid_ids."""
        return self._extract_valid_ids(context_data)
    
    def _validate_response(self, parsed_response: dict, raw_response: str) -> ValidationInfo:
        """Run validation rules against the parsed response."""
        context = {
            "valid_ids": self._valid_ids,
            "raw_response_length": len(raw_response)
        }
        
        validation_info = self.validator.validate(parsed_response, context, raw_response)
        
        return validation_info
    
    def _check_stopping_condition(self, validation_info: ValidationInfo, attempt: int) -> bool:
        """Check if agent should stop based on stopping conditions."""
        stop_on = self.stopping_conditions.get('stop_on', 'validated_output')
        max_attempts = self.stopping_conditions.get('max_attempts', 3)
        
        if attempt >= max_attempts:
            return True
        
        if stop_on == 'validated_output':
            return validation_info.is_valid
        elif stop_on == 'first_response':
            return True
        else:
            return validation_info.is_valid

    def _parse_json_response(self, llm_response) -> tuple[Optional[dict], Optional[str]]:
        """Parse LLM response to extract JSON."""
        raw_response = self._get_raw_response_text(llm_response)
        cleaned = raw_response.strip()
        
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            cleaned = '\n'.join(json_lines).strip()
        
        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError:
            pass
        
        candidates_tried = 0
        text = cleaned
        max_scan = min(len(text), 20000)
        i = 0
        while i < max_scan:
            if text[i] != '{':
                i += 1
                continue

            start = i
            depth = 0
            in_string = False
            escape = False

            for j in range(start, max_scan):
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        snippet = text[start:j + 1]
                        candidates_tried += 1
                        try:
                            return json.loads(snippet), None
                        except json.JSONDecodeError:
                            break

            i = start + 1
            if candidates_tried >= 25:
                break
        
        return None, "No valid JSON found in response"

    def _strip_unknown_values(self, data: Any) -> Any:
        """
        Recursively strip 'Unknown' values from parsed LLM output.
        
        - String "Unknown" (case-insensitive) → empty string ""
        - String "unknown" → empty string ""
        - String "N/A", "n/a", "None", "null" → empty string ""
        - Preserves structure (dicts, lists, nested values)
        
        This ensures the LLM doesn't pollute results with placeholder values.
        """
        unknown_values = {'unknown', 'n/a', 'none', 'null', 'not available', 'not specified'}
        
        if isinstance(data, dict):
            return {k: self._strip_unknown_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._strip_unknown_values(item) for item in data]
        elif isinstance(data, str):
            if data.lower().strip() in unknown_values:
                return ""
            return data
        else:
            return data

    def generate_config(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """DEPRECATED: Use execute() instead."""
        import warnings
        warnings.warn("generate_config() is deprecated. Use execute() instead.", DeprecationWarning)
        
        input_data = {
            "input_text": input_text
        }
        
        return self.execute(input_data=input_data, context=context)

    # Model defaults for different models
    model_defaults = {
        "gemini-2.5-flash-lite": "simple",
        "gemini-2.5-flash": "simple",
        "gemini-2.5-pro": "simple"
    }
