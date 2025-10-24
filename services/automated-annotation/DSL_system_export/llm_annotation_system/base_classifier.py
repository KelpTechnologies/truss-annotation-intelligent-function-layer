"""
LLM Annotation System - Lean Base Classifier (API-only mode)
===========================================================

A generalized LangChain-powered classifier for fashion product metadata annotation.
Supports structured outputs, prompt templating, and retry logic with VertexAI.
Modified for API-only operation - no local file dependencies.
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

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
import pandas as pd

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


class LLMAnnotationAgent:
    """
    LLM-powered classifier for fashion product metadata annotation.
    Modified for API-only operation.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        project_id: str = "truss-data-science",
        location: str = "us-central1",
        prompt_template_path: str = None,
        context_mode: str = "full-context",
        config: Optional[Dict[str, Any]] = None,
        log_IO: bool = False
    ):
        """
        Initialize the classifier agent.

        Args:
            model_name: VertexAI model to use
            project_id: GCP project ID for VertexAI
            location: GCP region
            prompt_template_path: Path to JSON prompt template file (API-loaded)
            context_mode: Context loading mode ('full-context', 'no-descriptions', 'reduced-taxonomy')
            config: Optional configuration dict to override defaults
            log_IO: Whether to log raw input/output data and JSON parsing results to CSV
        """
        # Apply configuration overrides
        if config:
            project_id = config.get('project_id', project_id)
            location = config.get('location', location)
            model_name = config.get('model_name', model_name)
            context_mode = config.get('context_mode', context_mode)

        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.context_mode = context_mode
        self.config = config or {}
        self.log_IO = log_IO

        # Initialize template config containers (will be populated if template is loaded)
        self.model_config_overrides = {}
        self.prompt_template = None

        # Load prompt template if provided (API-loaded templates)
        if prompt_template_path:
            self.load_prompt_template(prompt_template_path)

        # Initialize VertexAI model
        vertexai_kwargs = {
            "model_name": self.model_name,
            "project": self.project_id,
            "location": self.location,
            "temperature": self.config.get("temperature", 0.1),
            "max_output_tokens": self.config.get("max_output_tokens", 1024),
        }

        # Set model-specific defaults for better performance
        model_defaults = {
            "gemini-2.5-pro": {"temperature": 0.1, "max_output_tokens": 2048},
            "gemini-2.5-flash": {"temperature": 0.1, "max_output_tokens": 2048},
            "gemini-2.5-flash-lite": {"temperature": 0.1, "max_output_tokens": 2048},
        }

        if self.model_name in model_defaults:
            vertexai_kwargs.update(model_defaults[self.model_name])

        # Apply any model-specific overrides from template
        vertexai_kwargs.update(self.model_config_overrides)

        self.llm = ChatVertexAI(**vertexai_kwargs)

        # Initialize output parser for structured responses
        self.output_parser = PydanticOutputParser(pydantic_object=ClassificationResponse)

        logger.info(f"Initialized ChatVertexAI (multimodal) with model {self.model_name}")

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

    def prepare_multimodal_input(self, image_url: str, text_prompt: str, input_mode: str = "auto") -> List[Dict]:
        """
        Prepare multimodal input for VertexAI based on input mode.

        Args:
            image_url: URL of the image to analyze
            text_prompt: Text prompt to send to the model
            input_mode: 'image-only', 'text-only', 'multimodal', or 'auto'

        Returns:
            List of message dictionaries for ChatVertexAI
        """
        from langchain.schema import HumanMessage

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

        # Build message based on mode
        if effective_mode == "image-only":
            if not image_url:
                raise ValueError("Image URL required for image-only mode")

            # Download and encode image
            image_data = self._download_and_encode_image(image_url)

            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": text_prompt}
            ])

        elif effective_mode == "text-only":
            message = HumanMessage(content=text_prompt)

        elif effective_mode == "multimodal":
            if not image_url:
                raise ValueError("Image URL required for multimodal mode")

            # Download and encode image
            image_data = self._download_and_encode_image(image_url)

            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": text_prompt}
            ])
        else:
            raise ValueError(f"Unknown input mode: {effective_mode}")

        return [message]

    def _download_and_encode_image(self, image_url: str) -> str:
        """
        Download image from URL and return base64 encoded string.

        Args:
            image_url: URL of the image to download

        Returns:
            Base64 encoded image data
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

            # Encode to base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return image_data

        except Exception as e:
            raise ValueError(f"Failed to download/encode image from {image_url}: {e}")

    def classify(
        self,
        image_url: str = None,
        text_metadata: str = None,
        property_type: str = None,
        garment_id: str = None,
        root_type_id: int = None,
        brand: str = None,
        input_mode: str = "auto",
        context_data: Dict[str, Any] = None
    ) -> ClassificationResponse:
        """
        Classify a fashion item using multimodal LLM.

        Args:
            image_url: URL of the item image
            text_metadata: Additional text metadata (brand, title, description)
            property_type: Type of property to classify
            garment_id: Unique identifier for the item
            root_type_id: Root type ID for context filtering
            brand: Brand name for model filtering
            input_mode: Input mode ('auto', 'image-only', 'text-only', 'multimodal')
            context_data: Context data dictionary (API-loaded)

        Returns:
            ClassificationResponse with structured results
        """
        if not context_data:
            raise ValueError("Context data is required for API-only mode")

        if not property_type:
            raise ValueError("Property type is required")

        # Extract valid IDs from context data
        valid_ids = self._extract_valid_ids_from_context(context_data, property_type)

        if not valid_ids:
            logger.warning(f"No valid IDs found in context data for {property_type}")

        # Build prompt
        prompt_text = self._get_model_specific_prompt(
            context_data=context_data,
            model_name=self.model_name,
            text_metadata=text_metadata,
            input_mode=input_mode
        )

        # Prepare multimodal input
        multimodal_input = self.prepare_multimodal_input(
            image_url=image_url,
            text_prompt=prompt_text,
            input_mode=input_mode
        )

        # First attempt
        self._save_prompt_log(property_type, garment_id, prompt_text, input_mode, attempt=1, image_url=image_url)

        try:
            llm_response = self.llm.invoke(multimodal_input)
            raw_response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            result = self._robust_parse_response(
                llm_response,
                property_type,
                valid_ids,
                prompt_used=prompt_text,
                retry_count=0
            )

            # Log I/O data if enabled
            if self.log_IO:
                self._log_IO_data(
                    property_type=property_type,
                    garment_id=garment_id,
                    input_data={'image_url': image_url, 'text_metadata': text_metadata, 'input_mode': input_mode},
                    raw_prompt=prompt_text,
                    raw_response=raw_response_text,
                    parse_success=result is not None,
                    parsed_result=result
                )

            if result:
                self._save_prompt_log(property_type, garment_id, prompt_text, input_mode, result=result, attempt=1, image_url=image_url)
                logger.debug(f"Classification successful on attempt 1: {result.primary}")
                return result
            else:
                logger.warning(f"Parse failed on attempt 1, preparing retry...")

        except Exception as e:
            logger.error(f"LLM invocation failed on attempt 1: {e}")
            result = None

        # Retry attempt
        retry_prompt = self._build_format_nudge_prompt(prompt_text, property_type, valid_ids)
        self._save_prompt_log(property_type, garment_id, retry_prompt, input_mode, attempt=2, image_url=image_url)

        retry_input = self.prepare_multimodal_input(
            image_url=image_url,
            text_prompt=retry_prompt,
            input_mode=input_mode
        )

        try:
            llm_response = self.llm.invoke(retry_input)
            raw_response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            result = self._robust_parse_response(
                llm_response,
                property_type,
                valid_ids,
                prompt_used=retry_prompt,
                retry_count=1
            )

            # Log I/O data if enabled
            if self.log_IO:
                self._log_IO_data(
                    property_type=property_type,
                    garment_id=garment_id,
                    input_data={'image_url': image_url, 'text_metadata': text_metadata, 'input_mode': input_mode},
                    raw_prompt=retry_prompt,
                    raw_response=raw_response_text,
                    parse_success=result is not None,
                    parsed_result=result
                )

            if result:
                self._save_prompt_log(property_type, garment_id, retry_prompt, input_mode, result=result, attempt=2, image_url=image_url)
                logger.debug(f"Classification successful on attempt 2 (retry): {result.primary}")
                return result
            else:
                logger.error(f"Parse failed on retry attempt")

        except Exception as e:
            logger.error(f"LLM invocation failed on retry: {e}")

        # Both attempts failed - return unknown
        logger.error(f"All classification attempts failed for {property_type}. Returning unknown.")
        return ClassificationResponse(
            prediction_id=list(valid_ids)[0] if valid_ids else 0,  # Fallback to first valid ID
            scores=[PredictionScore(id=list(valid_ids)[0] if valid_ids else 0, score=0.0)]
        )

    def _extract_valid_ids_from_context(self, context_data: dict, property_type: str) -> set:
        """Extract valid taxonomy IDs from context data."""
        valid_ids = set()

        # Handle nested API response structure
        if isinstance(context_data, dict) and 'data' in context_data:
            context_data = context_data['data']

        # Context data contains items list with "ID X: name" format or structured data
        if isinstance(context_data, dict) and 'context_content' in context_data:
            # API-loaded context data format
            content = context_data['context_content']
            for item_id, item_data in content.items():
                try:
                    valid_ids.add(int(item_id))
                except (ValueError, TypeError):
                    continue
        else:
            # Legacy format - extract from items list
            items = context_data.get("materials", context_data.get("items", []))

            for item in items:
                # Extract ID from "ID X: name" format
                if isinstance(item, str) and item.startswith("ID "):
                    try:
                        id_part = item.split(":")[0].strip()  # "ID X"
                        id_num = int(id_part.split()[1])  # Extract X
                        valid_ids.add(id_num)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not extract ID from item '{item}': {e}")

        return valid_ids

    def _build_format_nudge_prompt(self, original_prompt: str, property_type: str, valid_ids: set) -> str:
        """Build a minimal format-only nudge prompt for retry."""
        nudge = (
            f"RETRY: Your previous response had formatting errors.\n\n"
            f"You MUST return ONLY valid JSON in this EXACT format:\n"
            "{\n"
            '  "prediction_id": <integer>,\n'
            '  "scores": [{"id": <int>, "score": <float>}, ...]\n'
            "}\n\n"
            f"Valid IDs for {property_type}: {sorted(list(valid_ids))}\n\n"
            f"NO explanations. NO markdown. ONLY the JSON object.\n\n"
            f"Original task:\n{original_prompt}"
        )
        return nudge

    def _get_model_specific_prompt(self, context_data, model_name, text_metadata=None, input_mode="image-only"):
        """Get model-specific prompt using modular template system with optional text metadata."""
        # Get the appropriate template for this model
        template_name = self.model_defaults.get(model_name, "standard")

        # Handle nested API response structure
        if isinstance(context_data, dict) and 'data' in context_data:
            context_data = context_data['data']

        # Build context string from context_data
        if isinstance(context_data, dict) and 'context_content' in context_data:
            # API-loaded context data format
            content = context_data['context_content']
            context_items = []

            for item_id, item_data in content.items():
                if isinstance(item_data, dict):
                    name = item_data.get('name', f'ID {item_id}')
                    description = item_data.get('description', '')

                    if self.context_mode == "no-descriptions":
                        context_items.append(f"ID {item_id}: {name}")
                    else:
                        context_items.append(f"ID {item_id}: {name} - {description}")

            context_str = "\n".join(context_items)

        elif isinstance(context_data, dict) and "materials" in context_data:
            # New structured context format - use items list with optional descriptions
            items = context_data["materials"]
            descriptions = context_data.get("descriptions", {})
            mode = context_data.get("mode", "")

            # Include descriptions for full-context and reduced-taxonomy modes
            if descriptions and mode in ["full-context", "reduced-taxonomy"]:
                context_str = "\n".join([f"ID {item_id}: {name} - {descriptions.get(str(item_id), '')}"
                                       for item_id, name in items.items()])
            else:
                # no-descriptions mode or missing descriptions
                context_str = "\n".join([f"ID {item_id}: {name}" for item_id, name in items.items()])

        elif isinstance(context_data, dict):
            # Dict without materials key - try to get context string
            context_str = context_data.get("context", "")
            if not isinstance(context_str, str):
                # Fallback: convert entire dict to string representation
                logger.warning(f"Context data is dict but not properly formatted: {type(context_str)}")
                context_str = str(context_data)
        else:
            # Assume it's already a string
            context_str = str(context_data) if context_data else ""

        # Build prompt from template components
        prompt_parts = []

        # Get template configuration
        if self.prompt_template and template_name in self.prompt_template.get('templates', {}):
            template = self.prompt_template['templates'][template_name]

            # Add system message
            if 'system_message' in template:
                prompt_parts.append(template['system_message'])

            # Add context
            if 'context_intro' in template:
                context_section = f"{template['context_intro']}\n{context_str}"
                prompt_parts.append(context_section)

            # Add text metadata if provided
            if text_metadata and input_mode in ["multimodal", "text-only"]:
                prompt_parts.append(f"Additional Product Information:\n{text_metadata}")

            # Add instructions
            if 'instructions' in template:
                prompt_parts.append(template['instructions'])

            # Add output format
            if 'output_format' in template:
                output_format = template['output_format']
                # Add reasoning field to output format if not present
                if '"reasoning"' not in output_format:
                    # Insert reasoning field before closing brace
                    output_format = output_format.rstrip('}').rstrip() + ',\n  "reasoning": "brief explanation under 30 words"\n}'
                prompt_parts.append(f"Output Format:\n{output_format}")

        else:
            # Fallback template if no modular template available
            prompt_parts = [
                "You are a fashion product classification expert.",
                f"Context:\n{context_str}",
            ]

            if text_metadata and input_mode in ["multimodal", "text-only"]:
                prompt_parts.append(f"Additional Product Information:\n{text_metadata}")

            prompt_parts.extend([
                "Analyze the image and classify the product.",
                'Return ONLY valid JSON: {"prediction_id": <integer>, "scores": [{"id": <int>, "score": <float>}], "reasoning": "brief explanation"}'
            ])

        return "\n\n".join(prompt_parts)

    def _robust_parse_response(self, llm_response, property_type: str, valid_ids: set,
                              prompt_used: str = "", retry_count: int = 0) -> Optional[ClassificationResponse]:
        """Robustly parse LLM response with multiple fallback strategies."""
        raw_response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # Clean the response
        cleaned_response = raw_response.strip()

        # Remove markdown code blocks if present
        if cleaned_response.startswith("```"):
            lines = cleaned_response.split('\n')
            # Find JSON content between ``` markers
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            if json_lines:
                cleaned_response = '\n'.join(json_lines).strip()

        # Strategy 1: Try direct JSON parsing
        try:
            parsed_data = json.loads(cleaned_response.strip())
            logger.debug(f"Successfully parsed response as JSON: {parsed_data}")
        except json.JSONDecodeError:
            # Strategy 2: Try parsing entire response or largest JSON object
            parsed_data = None

            # First, try parsing the entire cleaned response as JSON
            try:
                parsed_data = json.loads(cleaned_response.strip())
                logger.debug(f"Successfully parsed entire response as JSON: {parsed_data}")
            except json.JSONDecodeError:
                # If that fails, try to find the largest JSON object using greedy matching
                logger.debug("Full response not JSON, trying to extract JSON object")

                # Find JSON object (greedy match from first { to last })
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        parsed_data = json.loads(json_str)
                        logger.debug(f"Successfully parsed extracted JSON: {parsed_data}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parse failed: {e}")

        if not parsed_data:
            logger.warning(f"Failed to parse LLM response for {property_type}")
            return None

        # Validate required fields
        if 'prediction_id' not in parsed_data:
            logger.warning(f"Missing prediction_id in parsed response for {property_type}")
            return None

        prediction_id = parsed_data['prediction_id']

        # Validate prediction_id is in valid_ids (unless it's 0 for unknown)
        if prediction_id != 0 and prediction_id not in valid_ids:
            logger.warning(f"Invalid prediction_id {prediction_id} for {property_type}. Valid IDs: {sorted(valid_ids)}")
            return None

        # Handle scores
        scores_data = parsed_data.get('scores', [])
        if not scores_data:
            # Fallback: create a single score for the prediction_id
            scores_data = [{'id': prediction_id, 'score': 0.8}]

        # Convert to PredictionScore objects
        scores = []
        for score_item in scores_data[:5]:  # Limit to top 5
            if isinstance(score_item, dict) and 'id' in score_item and 'score' in score_item:
                try:
                    score_obj = PredictionScore(
                        id=int(score_item['id']),
                        score=float(score_item['score'])
                    )
                    scores.append(score_obj)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid score item: {score_item}, error: {e}")

        if not scores:
            logger.warning(f"No valid scores found for {property_type}")
            return None

        # Extract reasoning if present
        reasoning = parsed_data.get('reasoning')

        return ClassificationResponse(
            prediction_id=prediction_id,
            scores=scores,
            reasoning=reasoning
        )

    def _log_IO_data(self, property_type: str, garment_id: str, input_data: Dict[str, Any],
                     raw_prompt: str, raw_response: str, parse_success: bool,
                     parsed_result: Any = None, parse_error: str = None):
        """Log raw input/output data and JSON parsing results to CSV."""
        if not self.log_IO:
            return

        import csv
        from pathlib import Path

        io_logs_dir = Path("llm_io_logs")
        io_logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = io_logs_dir / f"llm_io_log_{property_type}_{timestamp}.csv"

        row = {
            'timestamp': datetime.now().isoformat(),
            'property_type': property_type,
            'garment_id': garment_id,
            'model_name': self.model_name,
            'input_mode': input_data.get('input_mode', 'unknown'),
            'has_image': bool(input_data.get('image_url')),
            'has_text': bool(input_data.get('text_metadata')),
            'image_url': input_data.get('image_url', ''),
            'text_metadata_length': len(input_data.get('text_metadata', '')),
            'root_type_id': input_data.get('root_type_id'),
            'brand': input_data.get('brand', ''),
            'raw_prompt_length': len(raw_prompt),
            'raw_response_length': len(raw_response),
            'raw_response_preview': raw_response[:500] + '...' if len(raw_response) > 500 else raw_response,
            'parse_success': parse_success,
            'parse_error': parse_error or '',
            'parsed_prediction_id': parsed_result.prediction_id if parsed_result and hasattr(parsed_result, 'prediction_id') else None,
            'parsed_primary': parsed_result.primary if parsed_result and hasattr(parsed_result, 'primary') else '',
            'parsed_confidence': parsed_result.confidence if parsed_result and hasattr(parsed_result, 'confidence') else None,
            'parsed_scores_count': len(parsed_result.scores) if parsed_result and hasattr(parsed_result, 'scores') else 0
        }

        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.debug(f"I/O data logged to: {csv_file}")

    def _save_prompt_log(self, property_type: str, garment_id: str, prompt_text: str,
                         input_mode: str, result: Any = None, attempt: int = 1, image_url: str = None):
        """
        Save complete prompt + metadata to log file for inspection.

        Args:
            property_type: Property being classified
            garment_id: Item identifier
            prompt_text: Complete prompt sent to LLM
            input_mode: Input mode used (image-only, text-only, multimodal)
            result: Classification result (if successful)
            attempt: Attempt number (1 or 2)
            image_url: URL of the image sent to the model (if applicable)
        """
        # File logging is disabled for API-only mode to avoid disk I/O issues
        # Logs are printed to terminal instead
        pass

    # Model defaults for different models
    model_defaults = {
        "gemini-2.5-flash-lite": "simple",
        "gemini-2.5-flash": "simple",
        "gemini-2.5-pro": "simple"
    }
