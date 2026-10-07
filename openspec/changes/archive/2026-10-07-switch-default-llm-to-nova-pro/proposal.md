# Change: Switch default LLM from Claude to Amazon Nova Pro

## Why

Anthropic Claude on Amazon Bedrock is billed through AWS Marketplace. This account's promotional credits do not cover it, and Claude is IAM-denied. Amazon Nova is a first-party Bedrock model and is credit-eligible.

## What Changes

- Default model becomes `us.amazon.nova-pro-v1:0` (us-east-1).
- Invocations use the Bedrock Converse API.
- IAM allows `foundation-model/amazon.nova-*` and `inference-profile/us.amazon.nova-*` instead of Anthropic.
- Model-id overrides remain, and `anthropic.*` ids are rejected.
- Cost tables, tests, and docs follow the new default.

## Impact

- Runtime: `lib/bedrock.py`, gateway, agents, resilience stack, fine-tune.
- Deploy: `template.yaml` parameters, secret payload, and IAM. Existing stacks must be redeployed with `BedrockModelId=us.amazon.nova-pro-v1:0`.
