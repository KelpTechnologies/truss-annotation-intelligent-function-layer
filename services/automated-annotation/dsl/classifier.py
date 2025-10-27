"""
Classifier facade using VertexAI + LangChain with structured response
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from io import BytesIO
import base64
import requests
from PIL import Image

from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionScore(BaseModel):
    id: int = Field(description="Taxonomy ID (integer)")
    score: float = Field(description="Raw model score (0-1)", ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    prediction_id: int = Field(description="Primary prediction taxonomy ID (integer)")
    scores: List[PredictionScore] = Field(min_items=1, max_items=5, description="Top-k predictions with raw scores")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation for the primary prediction (max 30 words)")

    @property
    def primary(self) -> str:
        return f"ID {self.prediction_id}"

    @property
    def alternatives(self) -> List[str]:
        return [f"ID {score.id}" for score in self.scores[1:]]

    @property
    def confidence(self) -> float:
        primary_score = next((s.score for s in self.scores if s.id == self.prediction_id), 0.0)
        return round(primary_score, 3)


class LLMAnnotationAgent:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        project_id: str = "truss-data-science",
        location: str = "us-central1",
        prompt_template_path: str = None,
        context_mode: str = "full-context",
        config: Optional[Dict[str, Any]] = None,
        log_IO: bool = False,
    ):
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

        self.model_config_overrides = {}
        self.prompt_template = None

        if prompt_template_path:
            self.load_prompt_template(prompt_template_path)

        vertexai_kwargs = {
            "model_name": self.model_name,
            "project": self.project_id,
            "location": self.location,
            "temperature": self.config.get("temperature", 0.1),
            "max_output_tokens": self.config.get("max_output_tokens", 1024),
        }
        model_defaults = {
            "gemini-2.5-pro": {"temperature": 0.1, "max_output_tokens": 2048},
            "gemini-2.5-flash": {"temperature": 0.1, "max_output_tokens": 2048},
            "gemini-2.5-flash-lite": {"temperature": 0.1, "max_output_tokens": 2048},
        }
        if self.model_name in model_defaults:
            vertexai_kwargs.update(model_defaults[self.model_name])
        vertexai_kwargs.update(self.model_config_overrides)
        self.llm = ChatVertexAI(**vertexai_kwargs)

        self.output_parser = PydanticOutputParser(pydantic_object=ClassificationResponse)
        logger.info(f"Initialized ChatVertexAI (multimodal) with model {self.model_name}")

    def load_prompt_template(self, template_path: str) -> None:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_config = json.load(f)
        self.prompt_template = template_config
        model_overrides = template_config.get('model_overrides', {})
        self.model_config_overrides = model_overrides.get(self.model_name, {})

    def prepare_multimodal_input(self, image_url: str, text_prompt: str, input_mode: str = "auto") -> List[Dict]:
        from langchain.schema import HumanMessage
        if input_mode == "auto":
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

        if effective_mode in ("image-only", "multimodal"):
            image_data = self._download_and_encode_image(image_url)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": text_prompt},
            ]
            return [HumanMessage(content=content)]
        if effective_mode == "text-only":
            return [HumanMessage(content=text_prompt)]
        raise ValueError(f"Unknown input mode: {effective_mode}")

    def _download_and_encode_image(self, image_url: str) -> str:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.verify()
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def classify(
        self,
        image_url: str = None,
        text_metadata: str = None,
        property_type: str = None,
        garment_id: str = None,
        root_type_id: int = None,
        brand: str = None,
        input_mode: str = "auto",
        context_data: Dict[str, Any] = None,
    ) -> ClassificationResponse:
        if not context_data:
            raise ValueError("Context data is required for API-only mode")
        if not property_type:
            raise ValueError("Property type is required")

        valid_ids = self._extract_valid_ids_from_context(context_data, property_type)
        prompt_text = self._get_model_specific_prompt(
            context_data=context_data,
            model_name=self.model_name,
            text_metadata=text_metadata,
            input_mode=input_mode,
        )
        multimodal_input = self.prepare_multimodal_input(
            image_url=image_url,
            text_prompt=prompt_text,
            input_mode=input_mode,
        )
        try:
            llm_response = self.llm.invoke(multimodal_input)
            result = self._robust_parse_response(llm_response, property_type, valid_ids, prompt_used=prompt_text, retry_count=0)
            if result:
                return result
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")

        retry_prompt = self._build_format_nudge_prompt(prompt_text, property_type, valid_ids)
        retry_input = self.prepare_multimodal_input(image_url=image_url, text_prompt=retry_prompt, input_mode=input_mode)
        try:
            llm_response = self.llm.invoke(retry_input)
            result = self._robust_parse_response(llm_response, property_type, valid_ids, prompt_used=retry_prompt, retry_count=1)
            if result:
                return result
        except Exception as e:
            logger.error(f"LLM invocation failed on retry: {e}")

        first_id = list(valid_ids)[0] if valid_ids else 0
        return ClassificationResponse(prediction_id=first_id, scores=[PredictionScore(id=first_id, score=0.0)])

    def _extract_valid_ids_from_context(self, context_data: dict, property_type: str) -> set:
        valid_ids = set()
        if isinstance(context_data, dict) and 'data' in context_data:
            context_data = context_data['data']
        if isinstance(context_data, dict) and 'context_content' in context_data:
            content = context_data['context_content']
            for item_id in content.keys():
                try:
                    valid_ids.add(int(item_id))
                except (ValueError, TypeError):
                    continue
        else:
            items = context_data.get("materials", context_data.get("items", [])) if isinstance(context_data, dict) else []
            for item in items:
                if isinstance(item, str) and item.startswith("ID "):
                    try:
                        id_num = int(item.split(":")[0].strip().split()[1])
                        valid_ids.add(id_num)
                    except (ValueError, IndexError):
                        continue
        return valid_ids

    def _build_format_nudge_prompt(self, original_prompt: str, property_type: str, valid_ids: set) -> str:
        return (
            "RETRY: Your previous response had formatting errors.\n\n"
            "You MUST return ONLY valid JSON in this EXACT format:\n"
            "{\n"
            '  "prediction_id": <integer>,\n'
            '  "scores": [{"id": <int>, "score": <float>}, ...]\n'
            "}\n\n"
            f"Valid IDs for {property_type}: {sorted(list(valid_ids))}\n\n"
            "NO explanations. NO markdown. ONLY the JSON object.\n\n"
            f"Original task:\n{original_prompt}"
        )

    def _get_model_specific_prompt(self, context_data, model_name, text_metadata=None, input_mode="image-only"):
        template_name = self.model_defaults.get(model_name, "standard")
        if isinstance(context_data, dict) and 'data' in context_data:
            context_data = context_data['data']

        if isinstance(context_data, dict) and 'context_content' in context_data:
            content = context_data['context_content']
            items = []
            for item_id, item_data in content.items():
                if isinstance(item_data, dict):
                    name = item_data.get('name', f'ID {item_id}')
                    desc = item_data.get('description', '')
                    items.append(f"ID {item_id}: {name} - {desc}" if self.context_mode != "no-descriptions" else f"ID {item_id}: {name}")
            context_str = "\n".join(items)
        elif isinstance(context_data, dict) and "materials" in context_data:
            items = context_data["materials"]
            descriptions = context_data.get("descriptions", {})
            mode = context_data.get("mode", "")
            if descriptions and mode in ["full-context", "reduced-taxonomy"]:
                context_str = "\n".join([f"ID {item_id}: {name} - {descriptions.get(str(item_id), '')}" for item_id, name in items.items()])
            else:
                context_str = "\n".join([f"ID {item_id}: {name}" for item_id, name in items.items()])
        elif isinstance(context_data, dict):
            context_str = context_data.get("context", "")
            if not isinstance(context_str, str):
                context_str = str(context_data)
        else:
            context_str = str(context_data) if context_data else ""

        parts = []
        if self.prompt_template and template_name in self.prompt_template.get('templates', {}):
            template = self.prompt_template['templates'][template_name]
            if 'system_message' in template:
                parts.append(template['system_message'])
            if 'context_intro' in template:
                parts.append(f"{template['context_intro']}\n{context_str}")
            if text_metadata and input_mode in ["multimodal", "text-only"]:
                parts.append(f"Additional Product Information:\n{text_metadata}")
            if 'instructions' in template:
                parts.append(template['instructions'])
            if 'output_format' in template:
                output_format = template['output_format']
                if '"reasoning"' not in output_format:
                    output_format = output_format.rstrip('}').rstrip() + ',\n  "reasoning": "brief explanation under 30 words"\n}'
                parts.append(f"Output Format:\n{output_format}")
        else:
            parts = [
                "You are a fashion product classification expert.",
                f"Context:\n{context_str}",
            ]
            if text_metadata and input_mode in ["multimodal", "text-only"]:
                parts.append(f"Additional Product Information:\n{text_metadata}")
            parts.extend([
                "Analyze the image and classify the product.",
                'Return ONLY valid JSON: {"prediction_id": <integer>, "scores": [{"id": <int>, "score": <float>}], "reasoning": "brief explanation"}'
            ])
        return "\n\n".join(parts)

    # Model defaults for different models
    model_defaults = {
        "gemini-2.5-flash-lite": "simple",
        "gemini-2.5-flash": "simple",
        "gemini-2.5-pro": "simple",
    }
