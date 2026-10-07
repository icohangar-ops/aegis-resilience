# Design: Amazon Nova Pro via Converse

## Decision

Call Bedrock through `converse` so Nova, Titan, and Llama share one request and response shape. The default model id is the US geo inference profile `us.amazon.nova-pro-v1:0`, which routes from us-east-1 to us-east-1, us-east-2, and us-west-2.

## IAM

Invoke permissions include:

- `arn:aws:bedrock:<region>::foundation-model/amazon.nova-*` in the stack region plus us-east-1, us-east-2, and us-west-2 (geo-profile destinations)
- `arn:aws:bedrock:<region>:<account>:inference-profile/us.amazon.nova-*`

Fallback Titan and tertiary Llama foundation-model ARNs stay on the invoke roles.

## Rejection

`assert_model_allowed` strips a leading `us.`, `eu.`, `apac.`, or `global.` prefix and rejects any id whose provider is `anthropic`. The SAM parameters use the same constraint so a stack update cannot select Claude.

## Cost

Nova Pro on-demand rates in us-east-1: $0.80 per 1M input tokens and $3.20 per 1M output tokens. Inference-profile ids resolve to the foundation-model price by stripping the geo prefix. Fine-tuning jobs submit the foundation-model id (`amazon.nova-pro-v1:0`), not the inference-profile id.
