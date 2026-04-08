# Agent Telemetry System

Unified agent execution logging emitting `AGENT_EXECUTION` log type to CloudWatch via the structured logger. Supports per-agent debug gating, multi-step workflow correlation, and optional local JSONL output for dev.

---

## Quick Reference

| What | How |
|------|-----|
| Enable debug for specific agents | `DEBUG_AGENTS=classifier-size-*` env var |
| Enable debug for all agents | `LOG_LEVEL=DEBUG` |
| Enable local JSONL logging | `ENABLE_LOCAL_LOGS=1` (dev only) |
| Find all steps for a request | `filter logType = "AGENT_EXECUTION" and requestId = "req_xxx"` |
| Find all steps in a workflow | `filter logType = "AGENT_EXECUTION" and workflowId = "abc123"` |
| Track a product across services | `filter resourceId = "product-123"` |

---

## Architecture

```
Lambda Handler (index.py)
  │  structured_logger.start_request(event) → req_ctx
  ▼
Orchestration Function (e.g. classifier_api_orchestration.py)
  │  timer = StepTimer()
  │  with timer.step("config_load"): ...
  │  with timer.step("image_download"): ...
  │  agent_result = agent.execute(input_data)
  │    └─ base_agent.py internally times: prompt_build, llm_call, json_parse, validation
  │  agent_telemetry.log_execution(req_ctx, agent_result, timing, config_id, ...)
  ▼
CloudWatch Log (JSON)
  logType: "AGENT_EXECUTION"
  + timing, tokens, validation, classification result
  + (DEBUG only) inputData, outputData, reasoning
```

### File Layout

| File | Role |
|------|------|
| `agent_telemetry.py` | `AgentTelemetry` class + `StepTimer` + `generate_workflow_id()` |
| `agent_architecture/base_agent.py` | Wraps execute() phases with StepTimer (prompt_build, llm_call, json_parse, validation) |
| `structured_logger.py` | Underlying `_emit()` method — writes JSON to stdout (CloudWatch) |

`agent_telemetry.py` lives in `automated-annotation/` only. It is NOT distributed via copy-utils — it's agent-specific, not shared with DSL.

---

## Log Schema: AGENT_EXECUTION

All fields below are emitted as a single JSON log line via `structured_logger._emit()`.

### Always present (from structured logger base)

| Field | Source | Example |
|-------|--------|---------|
| `schemaVersion` | structured_logger | `2.2` |
| `logType` | hardcoded | `"AGENT_EXECUTION"` |
| `ts` | structured_logger | `"2026-03-19T14:30:00.000Z"` |
| `requestId` | req_ctx | `"req_1710856200000_abc123def"` |
| `correlationId` | req_ctx | `"corr_1710856200000_xyz789"` |
| `resourceId` | req_ctx (v2.2) | `"product-abc-456"` |
| `resourceType` | req_ctx (v2.2) | `"annotation"` |
| `layer` | req_ctx | `"aifl"` |
| `serviceName` | req_ctx | `"automated-annotation"` |

### Agent execution fields

| Field | Type | Description |
|-------|------|-------------|
| `configId` | string | Agent config ID (e.g. `"classifier-colour-bags"`) |
| `model` | string | LLM model name (e.g. `"gemini-2.5-flash-lite"`) |
| `inputMode` | string | `"image-only"`, `"text-only"`, `"multimodal"`, `"auto"` |
| `status` | string | `"success"`, `"validation_failed"`, `"parsing_failed"`, `"llm_error"`, `"invalid_input"` |
| `attempt` | int | Attempt number (for retry logic) |

### Timing (nested object)

| Field | Type | Description |
|-------|------|-------------|
| `timing.configLoadMs` | float | DynamoDB config fetch (set in orchestration) |
| `timing.imageDownloadMs` | float | Image URL fetch + retry (set in orchestration) |
| `timing.promptBuildMs` | float | Prompt template + schema assembly (set in base_agent) |
| `timing.llmCallMs` | float | LLM API round-trip (set in base_agent) |
| `timing.jsonParseMs` | float | Response JSON extraction (set in base_agent) |
| `timing.validationMs` | float | Rule evaluation (set in base_agent) |
| `timing.bqSearchMs` | float | BigQuery lookup (set in orchestration, e.g. brand/size) |
| `timing.totalMs` | float | End-to-end for this step |

Orchestration adds outer steps (`config_load`, `image_download`, `bq_search`). Base agent adds inner steps (`prompt_build`, `llm_call`, `json_parse`, `validation`). Both contribute to the same timing dict via `StepTimer`.

### Token usage

| Field | Type | Description |
|-------|------|-------------|
| `tokensInput` | int | Prompt tokens |
| `tokensOutput` | int | Completion tokens |
| `tokensTotal` | int | Total tokens |

### LLM metadata

| Field | Type | Description |
|-------|------|-------------|
| `llmStopReason` | string | e.g. `"STOP"`, `"MAX_TOKENS"` |
| `llmTrafficType` | string | `"PRIORITY"` or `"STANDARD"` |
| `llmCallType` | string | `"multimodal"` or `"text"` |

### Validation & classification result

| Field | Type | Description |
|-------|------|-------------|
| `validationPassed` | bool | Whether validation rules passed |
| `validationCategory` | string | Validation failure category (e.g. `"low_confidence"`) |
| `confidence` | float | Primary prediction confidence (0-1) |
| `primaryId` | int | Predicted taxonomy ID |
| `primaryName` | string | Predicted taxonomy name |

### Error fields

| Field | Type | Description |
|-------|------|-------------|
| `error` | string/null | Error message (truncated to 500 chars) |
| `errorType` | string/null | Error classification (e.g. `"llm_error"`, `"validation_failed"`) |

### Test-only fields

| Field | Type | Description |
|-------|------|-------------|
| `expectedId` | any/null | Expected prediction ID. Always `null` in prod. Set by test framework. |

### Debug-gated fields (only when DEBUG_AGENTS matches or LOG_LEVEL=DEBUG)

| Field | Type | Description |
|-------|------|-------------|
| `inputData` | object | Full agent input (truncated to 5000 chars) |
| `outputData` | object | Full parsed LLM output (truncated to 5000 chars) |
| `reasoning` | string | LLM reasoning text |
| `scores` | string | Top-k prediction scores JSON |

### Workflow correlation

| Field | Type | Description |
|-------|------|-------------|
| `workflowId` | string/null | 12-char hex ID grouping multi-step orchestrations |
| `workflowStep` | string/null | Step label (e.g. `"brand_extraction"`, `"brand_classification"`) |

---

## Correlation ID Hierarchy

| ID | Scope | Set by | Use case |
|----|-------|--------|----------|
| `correlationId` | Cross-service (queue-processor → DSL → AIFL) | HTTP header / structured logger | Full product journey |
| `requestId` | Single Lambda invocation | structured logger | Everything in one API call |
| `workflowId` | Multi-step orchestration (e.g. brand: 3 steps) | orchestration function | Agent pipeline steps |
| `resourceId` | Product/entity being processed | v2.2 extraction from headers/path/body | Product-level queries |

---

## Environment Variables

### DEBUG_AGENTS

Comma-separated glob patterns matched against `configId`. When matched, full I/O is logged.

```bash
# Debug all size agents
DEBUG_AGENTS=classifier-size-*

# Debug size + brand agents
DEBUG_AGENTS=classifier-size-*,brand-*

# Debug one specific agent
DEBUG_AGENTS=classifier-colour-bags-text
```

Set in Lambda env vars or pass per-invocation. Only affects `inputData`, `outputData`, `reasoning`, `scores` fields.

### ENABLE_LOCAL_LOGS

When set to any truthy value, writes JSONL to `/tmp/logs/agent_telemetry.jsonl`. Never set in prod — Lambda `/tmp` is ephemeral and small (512MB).

### LOG_LEVEL

Standard log level gating. `DEBUG` acts as a blanket override for `DEBUG_AGENTS` (debugs all agents).

---

## Usage Examples

### Single-step orchestration (classifier_api_orchestration.py)

```python
from agent_telemetry import agent_telemetry, StepTimer

def classify_for_api(config_id, image_url, text_input, ...):
    timer = StepTimer()

    with timer.step("config_load"):
        config = config_loader.load_full_agent_config(config_id)

    with timer.step("image_download"):
        image_url = fetch_signed_url(image_id)

    agent = LLMAnnotationAgent(full_config=config)
    agent_result = agent.execute(input_data=input_data)

    # Merge orchestration timing with agent's internal timing
    timing = timer.finish()
    if agent_result.metadata.get("step_timing"):
        timing.update(agent_result.metadata["step_timing"])

    agent_telemetry.log_execution(
        req_ctx=req_ctx,
        agent_result=agent_result,
        timing=timing,
        config_id=config_id,
        input_data=input_data,
    )
```

### Multi-step workflow (brand_classification_orchestration.py)

```python
from agent_telemetry import agent_telemetry, StepTimer, generate_workflow_id

def run_brand_classification_workflow(raw_text, req_ctx=None, ...):
    workflow_id = generate_workflow_id()

    # Step 1: Brand extraction
    timer1 = StepTimer()
    with timer1.step("config_load"):
        config = load_config("brand-extraction-v1")
    agent1 = LLMAnnotationAgent(full_config=config)
    result1 = agent1.execute(input_data=...)

    timing1 = timer1.finish()
    if result1.metadata.get("step_timing"):
        timing1.update(result1.metadata["step_timing"])

    agent_telemetry.log_execution(
        req_ctx=req_ctx, agent_result=result1,
        timing=timing1, config_id="brand-extraction-v1",
        workflow_id=workflow_id, workflow_step="brand_extraction",
    )

    # Step 2: BigQuery search (non-agent step, still timed)
    timer_bq = StepTimer()
    with timer_bq.step("bq_search"):
        matches = search_brand_database(candidates)
    # Log as a workflow step without an agent result if needed

    # Step 3: Brand classification
    timer2 = StepTimer()
    with timer2.step("config_load"):
        config2 = load_config("brand-classification-v1")
    agent2 = LLMAnnotationAgent(full_config=config2)
    result2 = agent2.execute(input_data=...)

    timing2 = timer2.finish()
    if result2.metadata.get("step_timing"):
        timing2.update(result2.metadata["step_timing"])

    agent_telemetry.log_execution(
        req_ctx=req_ctx, agent_result=result2,
        timing=timing2, config_id="brand-classification-v1",
        workflow_id=workflow_id, workflow_step="brand_classification",
    )
```

---

## CloudWatch Insights Queries

```sql
-- Agent latency breakdown by config
fields configId, timing.llmCallMs, timing.promptBuildMs, timing.validationMs, timing.totalMs
| filter logType = "AGENT_EXECUTION"
| stats avg(timing.llmCallMs) as avg_llm, avg(timing.totalMs) as avg_total by configId

-- Token usage by model
fields model, tokensInput, tokensOutput, tokensTotal
| filter logType = "AGENT_EXECUTION"
| stats sum(tokensTotal) as total_tokens, avg(tokensTotal) as avg_tokens, count(*) as calls by model

-- Failure rate by config
fields configId, status
| filter logType = "AGENT_EXECUTION"
| stats sum(status != "success") as failures, count(*) as total by configId
| display configId, failures, total, (failures / total * 100) as failure_pct

-- Validation failure breakdown
fields configId, validationCategory, error
| filter logType = "AGENT_EXECUTION" and status = "validation_failed"
| stats count(*) by configId, validationCategory

-- Full agent trajectory for a product
fields @timestamp, logType, configId, status, timing.totalMs, confidence, primaryName
| filter resourceId = "product-abc-456" and logType = "AGENT_EXECUTION"
| sort @timestamp asc

-- Multi-step workflow trace
fields @timestamp, workflowStep, configId, status, timing.totalMs, tokensTotal
| filter logType = "AGENT_EXECUTION" and workflowId = "abc123def456"
| sort @timestamp asc

-- Slowest agent steps (p95)
fields configId, timing.llmCallMs, timing.totalMs
| filter logType = "AGENT_EXECUTION"
| stats pct(timing.llmCallMs, 95) as p95_llm, pct(timing.totalMs, 95) as p95_total by configId
| sort p95_total desc

-- Debug: full I/O for a specific request (requires DEBUG_AGENTS set)
fields configId, inputData, outputData, reasoning, timing.totalMs
| filter logType = "AGENT_EXECUTION" and requestId = "req_xxx" and inputData != ""
```

---

## StepTimer Reference

```python
from agent_telemetry import StepTimer

timer = StepTimer()

# Context manager — times the block
with timer.step("config_load"):
    config = load_config(...)

# Manual add — for pre-computed values
timer.add("image_download", 340.5)

# Finish — returns camelCase timing dict
timing = timer.finish()
# {"configLoadMs": 12.3, "imageDownloadMs": 340.5, "totalMs": 352.8}
```

**base_agent.py auto-timed steps:** `promptBuildMs`, `llmCallMs`, `jsonParseMs`, `validationMs`

These are stored in `agent_result.metadata["step_timing"]` and should be merged into the orchestration's timer output before calling `log_execution()`.

---

## Integration Checklist

For each orchestration function that calls agents:

1. Import: `from agent_telemetry import agent_telemetry, StepTimer, generate_workflow_id`
2. Create `StepTimer()` at function entry
3. Wrap config loading, image fetching, BQ lookups with `timer.step("name")`
4. After `agent.execute()`, merge agent's `step_timing` into orchestration timer
5. Call `agent_telemetry.log_execution(...)` with all required fields
6. For multi-step: generate `workflow_id` once, pass `workflow_step` per step
7. Pass `req_ctx` through from handler (add as parameter if not already present)

### Files requiring integration

| File | Type | Steps | workflow_id needed? |
|------|------|-------|-------------------|
| `classifier_api_orchestration.py` | Single agent | config_load, image_download | No |
| `brand_classification_orchestration.py` | 2 agents + BQ | config_load, bq_search (x2) | Yes (3 steps) |
| `model_size_classification_orchestration.py` | 2 agents + BQ | config_load, bq_search (x2) | Yes (4 steps) |
| `keyword_classifier_orchestration.py` | Single agent | config_load | No |
| `classifier_model_orchestration.py` | Single agent | config_load | No |
