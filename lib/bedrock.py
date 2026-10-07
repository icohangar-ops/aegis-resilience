"""
Aegis Resilience — Shared Bedrock Client Library

Production wrapper around the Amazon Bedrock Converse API. Converse uses one
request and response shape for every supported model, so callers do not build
provider-specific payloads.

Default model: Amazon Nova Pro via the US geo inference profile
``us.amazon.nova-pro-v1:0`` in us-east-1. A model-id override is allowed.
Anthropic Claude ids (``anthropic.*``, including geo profiles such as
``us.anthropic.*``) are rejected: Claude on Bedrock is billed through AWS
Marketplace and is not covered by promotional credits on this account.

FTR Compliance Notes:
- Model invocations name an explicit model id (no wildcard model ids at runtime)
- Token counting for cost governance
- Guardrails integration for content safety
- Invocation logging for audit trail
- Model-agnostic Converse calls so fallback models share one code path
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotocoreConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "us.amazon.nova-pro-v1:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TIMEOUT_MS = 30000

# Geo prefixes used by Bedrock cross-region inference profiles.
_GEO_PREFIXES = ("us.", "eu.", "apac.", "global.")
_PROVIDER_PREFIXES = ("amazon.", "meta.", "anthropic.", "cohere.", "mistral.", "ai21.")

# Approximate on-demand cost per 1K tokens (USD), us-east-1.
# Nova Pro: $0.80 / 1M input, $3.20 / 1M output.
# Nova Lite: $0.06 / 1M input, $0.24 / 1M output.
# Nova Micro: $0.035 / 1M input, $0.14 / 1M output.
# Nova Premier: $2.50 / 1M input, $12.50 / 1M output.
# Keys are foundation-model ids. Inference-profile ids (us.amazon.nova-*)
# resolve to the same row after the geo prefix is stripped.
MODEL_COSTS = {
    "amazon.nova-pro-v1:0": {"input_per_1k": 0.0008, "output_per_1k": 0.0032},
    "amazon.nova-lite-v1:0": {"input_per_1k": 0.00006, "output_per_1k": 0.00024},
    "amazon.nova-micro-v1:0": {"input_per_1k": 0.000035, "output_per_1k": 0.00014},
    "amazon.nova-premier-v1:0": {"input_per_1k": 0.0025, "output_per_1k": 0.0125},
    "amazon.titan-text-premier-v1:0": {"input_per_1k": 0.0008, "output_per_1k": 0.0016},
    "amazon.titan-text-express-v1:0": {"input_per_1k": 0.0004, "output_per_1k": 0.0008},
    "meta.llama3-70b-instruct-v1:0": {"input_per_1k": 0.00265, "output_per_1k": 0.0035},
    "meta.llama3-8b-instruct-v1:0": {"input_per_1k": 0.0006, "output_per_1k": 0.0009},
}


class UnsupportedModelError(ValueError):
    """Raised when a model id is an Anthropic Claude model."""


def canonical_model_id(model_id: str) -> str:
    """Strip a geo inference-profile prefix, leaving the foundation-model id."""
    raw = (model_id or "").strip()
    lower = raw.lower()
    for prefix in _GEO_PREFIXES:
        if lower.startswith(prefix):
            rest = raw[len(prefix):]
            if rest.lower().startswith(_PROVIDER_PREFIXES):
                return rest
            break
    return raw


def assert_model_allowed(model_id: str) -> str:
    """
    Accept a model-id override and reject Anthropic Claude ids.

    Rejects ``anthropic.*`` and geo profiles such as ``us.anthropic.*``.
    """
    if model_id is None or not str(model_id).strip():
        raise UnsupportedModelError("A model id is required.")
    raw = str(model_id).strip()
    normalized = canonical_model_id(raw).lower()
    if normalized.startswith("anthropic.") or "/anthropic." in normalized:
        raise UnsupportedModelError(
            f"Model id '{raw}' is not allowed. Anthropic Claude models (anthropic.*) "
            "are billed through AWS Marketplace and are not covered by promotional "
            "credits on this account. Use an Amazon Nova model id such as "
            f"{DEFAULT_MODEL_ID}."
        )
    return raw


def model_from_env(name: str, default: str = DEFAULT_MODEL_ID) -> str:
    """Read a model id from the environment, applying the Anthropic rejection."""
    return assert_model_allowed(os.environ.get(name, default))


class ModelFamily(Enum):
    """Supported Bedrock model families."""
    AMAZON_NOVA = "nova"
    AMAZON_TITAN = "amazon"
    META_LLAMA = "meta"
    AI21_JURASSIC = "ai21"
    COHERE_COMMAND = "cohere"
    MISTRAL = "mistral"
    ANTHROPIC_CLAUDE = "anthropic"
    UNKNOWN = "unknown"

    @classmethod
    def from_model_id(cls, model_id: str) -> "ModelFamily":
        """Detect model family from a model id or inference-profile id."""
        model_lower = canonical_model_id(model_id).lower()
        if model_lower.startswith("anthropic.") or "claude" in model_lower:
            return cls.ANTHROPIC_CLAUDE
        if "nova" in model_lower:
            return cls.AMAZON_NOVA
        if "titan" in model_lower:
            return cls.AMAZON_TITAN
        if "llama" in model_lower or model_lower.startswith("meta."):
            return cls.META_LLAMA
        if "jamba" in model_lower or model_lower.startswith("ai21."):
            return cls.AI21_JURASSIC
        if "command" in model_lower or model_lower.startswith("cohere."):
            return cls.COHERE_COMMAND
        if "mistral" in model_lower or "mixtral" in model_lower:
            return cls.MISTRAL
        return cls.UNKNOWN


@dataclass
class BedrockRequest:
    """Structured Bedrock invocation request."""
    model_id: str
    prompt: str
    system_prompt: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    stop_sequences: List[str] = field(default_factory=list)
    guardrail_id: Optional[str] = None
    guardrail_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate request parameters — FTR: Input validation."""
        self.model_id = assert_model_allowed(self.model_id)
        if not self.prompt:
            raise ValueError("prompt is required")
        if self.max_tokens < 1 or self.max_tokens > 8192:
            raise ValueError(f"max_tokens must be between 1 and 8192, got {self.max_tokens}")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")


@dataclass
class BedrockResponse:
    """Structured Bedrock invocation response with full metadata."""
    request_id: str
    model_id: str
    response_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    model_family: ModelFamily
    stop_reason: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.response_text)


class BedrockClient:
    """
    Production-grade Bedrock Runtime client.

    Invocations go through the Converse API so Nova, Titan, and Llama share
    one request and response shape. The default model is Amazon Nova Pro
    (``us.amazon.nova-pro-v1:0``).

    FTR Compliance:
    - Explicit model ids (no wildcards)
    - Timeout enforcement prevents hanging
    - Cost tracking for budget governance
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        secrets_manager_arn: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        default_model_id: str = DEFAULT_MODEL_ID,
    ):
        self.region_name = region_name or os.environ.get("AWS_REGION", DEFAULT_REGION)
        self.secrets_manager_arn = secrets_manager_arn
        self.timeout_ms = timeout_ms
        self.default_model_id = assert_model_allowed(default_model_id)
        self._client = None
        self._config = None
        self._invocation_log: List[Dict[str, Any]] = []

    @property
    def client(self) -> boto3.client:
        """Lazy-initialize Bedrock Runtime client."""
        if self._client is None:
            config = {
                "service_name": "bedrock-runtime",
                "region_name": self.region_name,
            }
            if self.timeout_ms:
                config["config"] = BotocoreConfig(
                    connect_timeout=self.timeout_ms / 1000,
                    read_timeout=self.timeout_ms / 1000,
                    retries={"max_attempts": 0},  # FTR: We handle retries in resilience stack
                )
            self._client = boto3.client(**config)
        return self._client

    def get_config(self) -> Dict[str, Any]:
        """Retrieve Bedrock configuration from Secrets Manager if configured."""
        if self._config is None and self.secrets_manager_arn:
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=self.secrets_manager_arn)
            self._config = json.loads(response["SecretString"])
        return self._config or {}

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text input.

        Uses a character-based heuristic (approximately 4 chars per token for English).
        FTR Note: This is an approximation. For exact counts, use the model's
        actual token counting (returned in response metadata).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate invocation cost in USD.

        FTR: Cost tracking enables budget governance and chargeback.
        """
        costs = MODEL_COSTS.get(
            canonical_model_id(model_id),
            {"input_per_1k": 0.001, "output_per_1k": 0.003},
        )
        return (input_tokens / 1000) * costs["input_per_1k"] + (output_tokens / 1000) * costs["output_per_1k"]

    def _converse_kwargs(self, request: BedrockRequest) -> Dict[str, Any]:
        """Build a model-agnostic Converse API request."""
        kwargs: Dict[str, Any] = {
            "modelId": request.model_id,
            "messages": [
                {"role": "user", "content": [{"text": request.prompt}]},
            ],
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
                "topP": request.top_p,
            },
        }
        if request.stop_sequences:
            kwargs["inferenceConfig"]["stopSequences"] = request.stop_sequences
        if request.system_prompt:
            kwargs["system"] = [{"text": request.system_prompt}]
        if request.guardrail_id and request.guardrail_version:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": request.guardrail_id,
                "guardrailVersion": request.guardrail_version,
            }
        return kwargs

    @staticmethod
    def _parse_converse_response(response_body: Dict[str, Any]) -> tuple:
        """
        Parse a Converse API response.

        Returns (text, input_tokens, output_tokens, stop_reason).
        """
        content = response_body.get("output", {}).get("message", {}).get("content", [])
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
        usage = response_body.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        stop_reason = response_body.get("stopReason", "end_turn")
        return text, input_tokens, output_tokens, stop_reason

    def invoke(self, request: BedrockRequest) -> BedrockResponse:
        """
        Invoke a Bedrock model through the Converse API.

        FTR Compliance:
        - Explicit model id
        - Timeout enforcement
        - Cost estimation
        - Invocation logging
        """
        start_time = time.monotonic()
        model_id = request.model_id

        try:
            assert_model_allowed(model_id)
            raw_response = self.client.converse(**self._converse_kwargs(request))

            response_text, input_tokens, output_tokens, stop_reason = self._parse_converse_response(
                raw_response
            )
            latency_ms = (time.monotonic() - start_time) * 1000
            cost_usd = self.estimate_cost(model_id, input_tokens, output_tokens)

            response = BedrockResponse(
                request_id=request.request_id,
                model_id=model_id,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost_usd, 6),
                model_family=ModelFamily.from_model_id(model_id),
                stop_reason=stop_reason,
            )

            self._log_invocation(request, response)
            return response

        except UnsupportedModelError:
            raise
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            response = BedrockResponse(
                request_id=request.request_id,
                model_id=model_id,
                response_text="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=round(latency_ms, 2),
                cost_usd=0.0,
                model_family=ModelFamily.from_model_id(model_id) if model_id else ModelFamily.UNKNOWN,
                error=str(e),
                error_type=type(e).__name__,
            )
            self._log_invocation(request, response)
            return response

    def _log_invocation(self, request: BedrockRequest, response: BedrockResponse) -> None:
        """Record invocation in internal log for audit trail."""
        log_entry = {
            "request_id": request.request_id,
            "model_id": request.model_id,
            "prompt_length": len(request.prompt),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "success": response.success,
            "error": response.error,
            "timestamp": response.timestamp,
        }
        self._invocation_log.append(log_entry)

        # Keep log bounded (last 100 invocations)
        if len(self._invocation_log) > 100:
            self._invocation_log = self._invocation_log[-100:]

    def get_invocation_log(self) -> List[Dict[str, Any]]:
        """Retrieve invocation log for analysis."""
        return list(self._invocation_log)

    def get_total_cost(self) -> float:
        """Calculate total cost across all invocations."""
        return sum(entry["cost_usd"] for entry in self._invocation_log)

    def get_average_latency(self) -> float:
        """Calculate average latency across all invocations."""
        if not self._invocation_log:
            return 0.0
        total = sum(entry["latency_ms"] for entry in self._invocation_log)
        return total / len(self._invocation_log)
