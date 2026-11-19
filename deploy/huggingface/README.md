---
title: Multimodal Image Text Classifier API
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal image + caption inference API

This Space serves the FastAPI backend for the portfolio's live multimodal classification demo.

The Dockerfile clones the public application source from:
https://github.com/dnagaraju271/multimodal-image-text-classification

## Required Space variables

- `HF_MODEL_REPO`: Hugging Face model repository containing `image_only.keras`, `text_only.keras`, `multimodal.keras`, and `text_vocabulary.json`.
- `ALLOWED_ORIGINS`: `https://dnagaraju271.github.io`

For a private model repository, add `HF_TOKEN` as a Space Secret with read access.

After the build, verify:
- `/api/health`
- `/api/models/status`

The second endpoint must report `ready: true` and all three models available before enabling the portfolio demo.
