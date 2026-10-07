"""
Aegis Resilience — Bedrock Client Tests

Tests for the shared Bedrock client library with mocked AWS services.
Invocations use the Converse API and default to Amazon Nova Pro.

FTR Compliance: All Bedrock invocations use explicit model ids.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from lib.bedrock import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    BedrockClient,
    BedrockRequest,
    ModelFamily,
    UnsupportedModelError,
    canonical_model_id,
)


NOVA_PRO = "us.amazon.nova-pro-v1:0"


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def mock_boto3():
    """Mock boto3 clients."""
    from unittest.mock import MagicMock, patch

    with patch("lib.bedrock.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client, mock_boto3


@pytest.fixture
def bedrock_client(mock_boto3):
    """Create a BedrockClient with mocked boto3."""
    return BedrockClient(region_name="us-east-1")


@pytest.fixture
def nova_request():
    """Standard Nova Pro request for testing."""
    return BedrockRequest(
        model_id=NOVA_PRO,
        prompt="Analyze Q3 cash flow",
        system_prompt="You are a CFO assistant",
        max_tokens=1024,
        temperature=0.7,
    )


@pytest.fixture
def nova_response_data():
    """Standard Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Based on Q3 data, cash flow improved by 15%."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
    }


def _stub_converse(mock_client, payload):
    mock_client.converse.return_value = payload


# =========================================================================
# Defaults
# =========================================================================
class TestDefaults:
    def test_default_model_is_nova_pro(self):
        assert DEFAULT_MODEL_ID == "us.amazon.nova-pro-v1:0"
        assert DEFAULT_REGION == "us-east-1"

    def test_client_default_model(self, bedrock_client):
        assert bedrock_client.default_model_id == NOVA_PRO
        assert bedrock_client.region_name == "us-east-1"


# =========================================================================
# Model Family Detection Tests
# =========================================================================
class TestModelFamily:
    """Tests for model family detection from model ID."""

    def test_detect_amazon_nova(self):
        assert ModelFamily.from_model_id("us.amazon.nova-pro-v1:0") == ModelFamily.AMAZON_NOVA
        assert ModelFamily.from_model_id("amazon.nova-lite-v1:0") == ModelFamily.AMAZON_NOVA
        assert ModelFamily.from_model_id("amazon.nova-micro-v1:0") == ModelFamily.AMAZON_NOVA

    def test_nova_is_not_classified_as_titan(self):
        assert ModelFamily.from_model_id("amazon.nova-pro-v1:0") != ModelFamily.AMAZON_TITAN

    def test_detect_amazon_titan(self):
        assert ModelFamily.from_model_id("amazon.titan-text-premier-v1:0") == ModelFamily.AMAZON_TITAN
        assert ModelFamily.from_model_id("amazon.titan-text-express-v1:0") == ModelFamily.AMAZON_TITAN

    def test_detect_meta_llama(self):
        assert ModelFamily.from_model_id("meta.llama3-70b-instruct-v1:0") == ModelFamily.META_LLAMA
        assert ModelFamily.from_model_id("meta.llama3-8b-instruct-v1:0") == ModelFamily.META_LLAMA

    def test_detect_anthropic_for_rejection(self):
        assert ModelFamily.from_model_id("anthropic.claude-3-5-sonnet-20241022-v1:0") == ModelFamily.ANTHROPIC_CLAUDE
        assert ModelFamily.from_model_id("us.anthropic.claude-3-5-sonnet-20241022-v2:0") == ModelFamily.ANTHROPIC_CLAUDE

    def test_detect_unknown(self):
        assert ModelFamily.from_model_id("unknown.model.v1") == ModelFamily.UNKNOWN


# =========================================================================
# Model id policy
# =========================================================================
class TestModelPolicy:
    def test_override_nova_lite_allowed(self):
        req = BedrockRequest(model_id="us.amazon.nova-lite-v1:0", prompt="test")
        assert req.model_id == "us.amazon.nova-lite-v1:0"

    def test_reject_anthropic_model_id(self):
        with pytest.raises(UnsupportedModelError, match="not allowed"):
            BedrockRequest(
                model_id="anthropic.claude-3-5-sonnet-20241022-v1:0",
                prompt="test",
            )

    def test_reject_geo_anthropic_profile(self):
        with pytest.raises(UnsupportedModelError, match="anthropic"):
            BedrockRequest(
                model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                prompt="test",
            )

    def test_canonical_strips_geo_prefix(self):
        assert canonical_model_id("us.amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0"


# =========================================================================
# BedrockRequest Tests
# =========================================================================
class TestBedrockRequest:
    """Tests for request validation."""

    def test_valid_request(self):
        req = BedrockRequest(model_id=NOVA_PRO, prompt="test")
        assert req.prompt == "test"
        assert req.request_id is not None
        assert len(req.request_id) > 0

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt is required"):
            BedrockRequest(model_id="amazon.nova-pro-v1:0", prompt="")

    def test_max_tokens_validation(self):
        with pytest.raises(ValueError, match="max_tokens"):
            BedrockRequest(model_id="amazon.nova-pro-v1:0", prompt="test", max_tokens=99999)

    def test_temperature_validation(self):
        with pytest.raises(ValueError, match="temperature"):
            BedrockRequest(model_id="amazon.nova-pro-v1:0", prompt="test", temperature=3.0)


# =========================================================================
# BedrockClient Invoke Tests
# =========================================================================
class TestBedrockClientInvoke:
    """Tests for Bedrock model invocation via Converse."""

    def test_invoke_nova_success(self, bedrock_client, nova_request, nova_response_data, mock_boto3):
        """Successful Nova invocation should return a structured Converse response."""
        mock_client = mock_boto3[0]
        _stub_converse(mock_client, nova_response_data)

        response = bedrock_client.invoke(nova_request)

        assert response.success is True
        assert response.response_text == "Based on Q3 data, cash flow improved by 15%."
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model_family == ModelFamily.AMAZON_NOVA
        assert response.stop_reason == "end_turn"

        kwargs = mock_client.converse.call_args.kwargs
        assert kwargs["modelId"] == NOVA_PRO
        assert kwargs["system"] == [{"text": "You are a CFO assistant"}]
        assert kwargs["inferenceConfig"]["maxTokens"] == 1024
        assert "anthropic_version" not in json.dumps(kwargs)
        mock_client.invoke_model.assert_not_called()

    def test_invoke_titan_uses_converse(self, mock_boto3, nova_response_data):
        """Fallback models share the Converse response shape."""
        mock_client = mock_boto3[0]
        client = BedrockClient(region_name="us-east-1")
        titan_payload = dict(nova_response_data)
        titan_payload["output"] = {
            "message": {"role": "assistant", "content": [{"text": "Titan response"}]}
        }
        _stub_converse(mock_client, titan_payload)

        req = BedrockRequest(model_id="amazon.titan-text-premier-v1:0", prompt="test")
        response = client.invoke(req)

        assert response.success is True
        assert response.response_text == "Titan response"
        assert response.model_family == ModelFamily.AMAZON_TITAN
        assert mock_client.converse.call_args.kwargs["modelId"] == "amazon.titan-text-premier-v1:0"

    def test_invoke_error_returns_error_response(self, bedrock_client, nova_request, mock_boto3):
        """Failed invocation should return error response, not raise."""
        mock_client = mock_boto3[0]
        mock_client.converse.side_effect = Exception("ServiceUnavailable")

        response = bedrock_client.invoke(nova_request)

        assert response.success is False
        assert response.error is not None
        assert "ServiceUnavailable" in response.error
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_invocation_logging(self, bedrock_client, nova_request, nova_response_data, mock_boto3):
        """Each invocation should be logged."""
        mock_client = mock_boto3[0]
        _stub_converse(mock_client, nova_response_data)

        bedrock_client.invoke(nova_request)

        log = bedrock_client.get_invocation_log()
        assert len(log) == 1
        assert log[0]["success"] is True
        assert log[0]["model_id"] == NOVA_PRO


# =========================================================================
# Cost Estimation Tests
# =========================================================================
class TestCostEstimation:
    """Tests for cost estimation."""

    def test_nova_pro_cost(self):
        client = BedrockClient(region_name="us-east-1")
        cost = client.estimate_cost("us.amazon.nova-pro-v1:0", 1000, 500)
        # input: 1K * $0.0008 = $0.0008, output: 0.5K * $0.0032 = $0.0016
        assert abs(cost - 0.0024) < 0.00001

    def test_nova_profile_matches_foundation_price(self):
        client = BedrockClient(region_name="us-east-1")
        profile = client.estimate_cost("us.amazon.nova-pro-v1:0", 1000, 500)
        foundation = client.estimate_cost("amazon.nova-pro-v1:0", 1000, 500)
        assert profile == foundation

    def test_titan_cost(self):
        client = BedrockClient(region_name="us-east-1")
        cost = client.estimate_cost("amazon.titan-text-premier-v1:0", 1000, 500)
        # input: 1K * $0.0008 = $0.0008, output: 0.5K * $0.0016 = $0.0008
        assert abs(cost - 0.0016) < 0.0001

    def test_total_cost_tracking(self, bedrock_client, nova_request, nova_response_data, mock_boto3):
        """Total cost should accumulate across invocations."""
        mock_client = mock_boto3[0]
        _stub_converse(mock_client, nova_response_data)

        bedrock_client.invoke(nova_request)
        bedrock_client.invoke(nova_request)

        total_cost = bedrock_client.get_total_cost()
        assert total_cost > 0

    def test_average_latency(self, bedrock_client, nova_request, nova_response_data, mock_boto3):
        """Average latency should be computed correctly."""
        mock_client = mock_boto3[0]
        _stub_converse(mock_client, nova_response_data)

        bedrock_client.invoke(nova_request)

        avg_lat = bedrock_client.get_average_latency()
        assert avg_lat >= 0


# =========================================================================
# Token Estimation Tests
# =========================================================================
class TestTokenEstimation:
    """Tests for token estimation."""

    def test_empty_string(self):
        client = BedrockClient(region_name="us-east-1")
        assert client.estimate_tokens("") == 0

    def test_short_text(self):
        client = BedrockClient(region_name="us-east-1")
        tokens = client.estimate_tokens("Hello world")
        assert tokens >= 1

    def test_longer_text(self):
        client = BedrockClient(region_name="us-east-1")
        text = "word " * 400
        tokens = client.estimate_tokens(text)
        assert tokens >= 50  # ~200 chars / 4 per token


# =========================================================================
# Source guard: nothing in the app defaults to Claude
# =========================================================================
class TestNoClaudeDefaults:
    def test_production_sources_do_not_default_to_claude(self):
        root = Path(__file__).resolve().parents[1]
        needles = ("anthropic.claude", "anthic.claude", "anthropic_version")
        skip_dirs = {".git", "tests", "__pycache__", ".venv", "venv", "node_modules"}
        hits = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".yaml", ".yml", ".md", ".toml"}:
                continue
            if skip_dirs.intersection(path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    hits.append(f"{path.relative_to(root)} contains {needle}")
        assert hits == []

    def test_template_defaults_to_nova_pro(self):
        template = (Path(__file__).resolve().parents[1] / "template.yaml").read_text(encoding="utf-8")
        assert "Default: us.amazon.nova-pro-v1:0" in template
        assert "foundation-model/amazon.nova-*" in template
        assert "inference-profile/us.amazon.nova-*" in template
