# Hugging Face Authentication Setup

## Step 1: Request Access to Llama Models

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Request access to this model" button
3. Fill out the form and wait for approval (usually takes a few hours to days)

## Step 2: Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it something like "llm-training"
4. Set permissions to "Read"
5. Copy the token

## Step 3: Local Development Setup

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

Or set environment variable:
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## Step 4: GitHub Actions Setup

1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Name: `HUGGINGFACE_TOKEN`
5. Value: Your Hugging Face token
6. Save

## Alternative: Use Non-Gated Models

If you want to test without authentication, you can use these alternatives:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (smaller Llama-style model)
- `microsoft/DialoGPT-medium` (different architecture, needs code changes)
- `google/flan-t5-base` (encoder-decoder, needs code changes)
