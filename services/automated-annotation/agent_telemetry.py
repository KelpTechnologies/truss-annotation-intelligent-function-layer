"""
Agent Telemetry - Unified agent execution logging for CloudWatch + local dev.

Emits AGENT_EXECUTION log type via structured_logger._emit() for CloudWatch,
and optionally writes JSONL to /tmp/logs/ for local development.

Usage:
    from agent_telemetry import agent_telemetry, StepTimer, generate_workflow_id

    # Single-step orchestration
    timer = StepTimer()
    with timer.step("config_load"):
        config = load_config(config_id)
    with timer.step("image_download"):
        image = fetch_image(url)

    agent_result = agent.execute(input_data)

    agent_telemetry.log_execution(
        req_ctx=req_ctx,
        agent_result=agent_result,
        timing=timer.finish(),
        config_id=config_id,
        input_data=input_data,
    )

    # Multi-step workflow
    workflow_id = generate_workflow_id()
    agent_telemetry.log_execution(..., workflow_id=workflow_id, workflow_step="brand_extraction")
    agent_telemetry.log_execution(..., workflow_id=workflow_id, workflow_step="brand_classification")

Environment variables:
    DEBUG_AGENTS      - Glob pattern for config_ids to log full I/O (e.g. "classifier-size-*")
    ENABLE_LOCAL_LOGS - Write JSONL to /tmp/logs/ (dev only, never set in prod)
    LOG_LEVEL         - Standard log level gating (DEBUG, INFO, WARN, ERROR)
"""

import fnmatch
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# StepTimer — collects timing for each phase of agent execution
# ---------------------------------------------------------------------------

class StepTimer:
    """
    Collects per-step timing for agent execution phases.

    Usage:
        timer = StepTimer()
        with timer.step("config_load"):
            config = load_config(...)
        with timer.step("llm_call"):
            result = call_llm(...)
        timing = timer.finish()
        # timing = {"configLoadMs": 12, "llmCallMs": 1200, "totalMs": 1212}
    """

    def __init__(self):
        self._start = time.time()
        self._steps: Dict[str, float] = {}
        self._current_step: Optional[str] = None
        self._step_start: float = 0.0

    @contextmanager
    def step(self, name: str):
        """Time a named step. Name should be snake_case (converted to camelCase in output)."""
        step_start = time.time()
        try:
            yield
        finally:
            elapsed_ms = round((time.time() - step_start) * 1000, 1)
            self._steps[name] = elapsed_ms

    def add(self, name: str, elapsed_ms: float):
        """Manually add a timing entry (e.g. from agent metadata)."""
        self._steps[name] = round(elapsed_ms, 1)

    def finish(self) -> Dict[str, float]:
        """
        Return timing dict with camelCase keys + totalMs.

        snake_case step names are converted: "config_load" -> "configLoadMs"
        """
        total_ms = round((time.time() - self._start) * 1000, 1)
        result = {}
        for name, ms in self._steps.items():
            camel = self._to_camel(name) + "Ms"
            result[camel] = ms
        result["totalMs"] = total_ms
        return result

    @staticmethod
    def _to_camel(snake: str) -> str:
        parts = snake.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])


# ---------------------------------------------------------------------------
# Workflow ID generation
# ---------------------------------------------------------------------------

def generate_workflow_id() -> str:
    """Generate a 12-char hex workflow ID for grouping multi-step orchestrations."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# AgentTelemetry — main telemetry class
# ---------------------------------------------------------------------------

class AgentTelemetry:
    """
    Agent execution telemetry via composition over StructuredLogger.

    Emits AGENT_EXECUTION log type to CloudWatch. Optionally writes JSONL
    for local dev when ENABLE_LOCAL_LOGS=1.
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self._logger = logger
        self._local_log_dir = "/tmp/logs"
        self._local_file = None

    @property
    def logger(self) -> StructuredLogger:
        if self._logger is None:
            self._logger = StructuredLogger(layer="aifl", service_name="automated-annotation")
        return self._logger

    # -------------------------------------------------------------------
    # Main logging method
    # -------------------------------------------------------------------

    def log_execution(
        self,
        req_ctx: dict,
        agent_result,
        timing: Dict[str, float],
        config_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        workflow_step: Optional[str] = None,
        expected_id: Optional[Any] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single agent execution step.

        Args:
            req_ctx: Request context from structured_logger.start_request()
            agent_result: AgentResult from base_agent.execute()
            timing: Dict from StepTimer.finish() with per-step timing
            config_id: Agent config ID (e.g. "classifier-colour-bags")
            input_data: Agent input dict (logged only if DEBUG_AGENTS matches)
            workflow_id: Groups multi-step orchestrations (auto-generated in orchestration)
            workflow_step: Label for this step (e.g. "brand_extraction", "brand_classification")
            expected_id: Expected prediction ID for test runs (null in prod)
            extras: Additional context dict
        """
        meta = agent_result.metadata if agent_result else {}

        # Core fields
        record = {
            "configId": config_id,
            "model": meta.get("model"),
            "inputMode": meta.get("input_mode"),
            "status": agent_result.status.value if agent_result else "unknown",
            "timing": timing,
        }

        # Workflow correlation
        if workflow_id:
            record["workflowId"] = workflow_id
        if workflow_step:
            record["workflowStep"] = workflow_step

        # Token usage
        for key, field in [
            ("llm_tokens_input", "tokensInput"),
            ("llm_tokens_output", "tokensOutput"),
            ("llm_tokens_total", "tokensTotal"),
        ]:
            if meta.get(key) is not None:
                record[field] = meta[key]

        # LLM metadata
        if meta.get("llm_stop_reason"):
            record["llmStopReason"] = meta["llm_stop_reason"]
        if meta.get("llm_traffic_type"):
            record["llmTrafficType"] = meta["llm_traffic_type"]
        if meta.get("llm_call_type"):
            record["llmCallType"] = meta["llm_call_type"]

        # Validation
        record["validationPassed"] = meta.get("validation_valid", False)
        if meta.get("validation_category"):
            record["validationCategory"] = meta["validation_category"]

        # Classification result
        if meta.get("confidence") is not None:
            record["confidence"] = meta["confidence"]
        if meta.get("prediction_id") is not None:
            record["primaryId"] = meta["prediction_id"]
        if meta.get("prediction_name"):
            record["primaryName"] = meta["prediction_name"]

        # Attempt count
        record["attempt"] = meta.get("attempt", 1)

        # Error details
        if agent_result and agent_result.error_report:
            record["error"] = agent_result.error_report.message[:500] if agent_result.error_report.message else None
            record["errorType"] = agent_result.error_report.error_type
        else:
            record["error"] = None
            record["errorType"] = None

        # Accuracy tracking (null in prod, set in test runs)
        record["expectedId"] = expected_id

        # Debug-gated fields: full I/O + reasoning
        if self._should_debug(config_id):
            if input_data:
                record["inputData"] = self._safe_serialize(input_data, max_chars=5000)
            if agent_result and agent_result.result:
                record["outputData"] = self._safe_serialize(agent_result.result, max_chars=5000)
            if meta.get("reasoning"):
                record["reasoning"] = meta["reasoning"]
            if meta.get("scores"):
                record["scores"] = meta["scores"]

        # Extras
        if extras:
            record["extras"] = extras

        # Emit to CloudWatch via structured logger
        self.logger._emit("AGENT_EXECUTION", req_ctx, record)

        # Optionally write local JSONL
        if os.environ.get("ENABLE_LOCAL_LOGS"):
            self._write_jsonl(record, req_ctx)

    # -------------------------------------------------------------------
    # DEBUG_AGENTS gating
    # -------------------------------------------------------------------

    def _should_debug(self, config_id: str) -> bool:
        """
        Check if this config_id should have full I/O logged.

        Matches against DEBUG_AGENTS env var (comma-separated glob patterns).
        Also returns True if LOG_LEVEL=DEBUG (blanket debug).

        Examples:
            DEBUG_AGENTS=classifier-size-*          → debug all size agents
            DEBUG_AGENTS=classifier-size-*,brand-*  → debug size + brand agents
            LOG_LEVEL=DEBUG                         → debug everything
        """
        # Blanket debug
        if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
            return True

        debug_agents = os.environ.get("DEBUG_AGENTS", "")
        if not debug_agents:
            return False

        patterns = [p.strip() for p in debug_agents.split(",") if p.strip()]
        return any(fnmatch.fnmatch(config_id, pattern) for pattern in patterns)

    # -------------------------------------------------------------------
    # Local JSONL writing (dev only)
    # -------------------------------------------------------------------

    def _write_jsonl(self, record: dict, req_ctx: dict):
        """Append record to local JSONL file. Only called when ENABLE_LOCAL_LOGS=1."""
        try:
            os.makedirs(self._local_log_dir, exist_ok=True)
            filepath = os.path.join(self._local_log_dir, f"agent_telemetry.jsonl")
            flat = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "requestId": req_ctx.get("request_id"),
                "correlationId": req_ctx.get("correlation_id"),
                **record,
            }
            with open(filepath, "a") as f:
                f.write(json.dumps(flat, default=str) + "\n")
        except Exception:
            pass  # Never fail the request for logging

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _safe_serialize(data: Any, max_chars: int = 5000) -> Any:
        """Serialize data for logging, truncating if needed."""
        try:
            serialized = json.dumps(data, default=str)
            if len(serialized) > max_chars:
                return json.loads(serialized[:max_chars] + '..."')
            return data
        except (TypeError, ValueError):
            s = str(data)
            return s[:max_chars] if len(s) > max_chars else s


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

agent_telemetry = AgentTelemetry()
