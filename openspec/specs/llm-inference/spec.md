# llm-inference Specification

## Purpose

Defines how Aegis selects and calls its large language model on Amazon Bedrock.

## Requirements

### Requirement: Default model is Amazon Nova Pro

The system SHALL call Amazon Nova Pro with model id `us.amazon.nova-pro-v1:0` in `us-east-1` when no model-id override is configured. The invocation SHALL use the Bedrock Converse API.

#### Scenario: No override

- **WHEN** `PRIMARY_MODEL_ID` and `BedrockModelId` are left at their defaults
- **THEN** the runtime calls `bedrock-runtime.converse` with model id `us.amazon.nova-pro-v1:0`

### Requirement: Model-id override is allowed

The system SHALL accept a non-Anthropic model id from the `BedrockModelId` stack parameter, the `PRIMARY_MODEL_ID` environment variable, or a request `model_preference`.

#### Scenario: Nova Lite override

- **WHEN** a caller sets the model id to `us.amazon.nova-lite-v1:0`
- **THEN** that id is used for the Converse call

### Requirement: Anthropic model ids are rejected

The system SHALL reject model ids whose provider is `anthropic`, including geo inference profiles such as `us.anthropic.*`, with an error that states Anthropic models are not allowed and names `us.amazon.nova-pro-v1:0` as the replacement.

#### Scenario: Direct Anthropic id

- **WHEN** a caller supplies a model id beginning with `anthropic.`
- **THEN** the call fails with `UnsupportedModelError` and Bedrock is not invoked

#### Scenario: Geo Anthropic profile

- **WHEN** a caller supplies a model id beginning with `us.anthropic.`
- **THEN** the call fails with the same rejection

### Requirement: Nova IAM and cost

The deployment SHALL grant `bedrock:InvokeModel` on `foundation-model/amazon.nova-*` and on inference profile `us.amazon.nova-*`. Cost estimates for Nova Pro SHALL use $0.0008 per 1K input tokens and $0.0032 per 1K output tokens.
