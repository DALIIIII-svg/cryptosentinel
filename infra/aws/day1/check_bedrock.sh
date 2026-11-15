#!/usr/bin/env bash
# Day-1 Check B: is Amazon Bedrock usable in this Learner Lab?
# Run on your laptop (WSL) AFTER copying the Learner Lab credentials into ~/.aws/credentials.
#   bash infra/aws/day1/check_bedrock.sh
set -u
REGION="${AWS_REGION:-us-east-1}"

echo "== Identity =="
aws sts get-caller-identity --output text || { echo "[FAIL] AWS CLI not configured"; exit 1; }

echo "== Models visible in $REGION (Nova, Claude, Llama, Mistral) =="
aws bedrock list-foundation-models --region "$REGION" \
  --query "modelSummaries[].modelId" --output text 2>&1 \
  | tr '\t' '\n' | grep -i -E "nova|claude|llama|mistral" | head -30 \
  || echo "  [FAIL] cannot list models (Bedrock probably blocked)"

try_model () {
  local model="$1"
  echo "== Test call: $model =="
  if out=$(aws bedrock-runtime converse --region "$REGION" --model-id "$model" \
        --messages '[{"role":"user","content":[{"text":"Reply with the single word OK"}]}]' \
        --query "output.message.content[0].text" --output text 2>&1); then
    echo "  [PASS] answer: $out"
    return 0
  fi
  echo "  [FAIL] $out" | head -3
  return 1
}

try_model "us.amazon.nova-lite-v1:0" || try_model "amazon.nova-lite-v1:0" \
  || echo "=> Bedrock not usable: set LLM_PROVIDER=groq in .env.aws"
