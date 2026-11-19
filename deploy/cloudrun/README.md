# Deploy the inference API to Google Cloud Run

This deployment keeps the browser UI on GitHub Pages and runs FastAPI + TensorFlow on Cloud Run. Trained weights stay in the private Hugging Face model repository and are downloaded by the API at runtime.

## Required settings

- Region: `europe-west9` (Paris)
- Service: `multimodal-api`
- CPU: 2
- Memory: start with 4 GiB; increase only if logs show out-of-memory failures
- Request concurrency: 1
- Minimum instances: 0
- Maximum instances: 1
- Request timeout: 300 seconds
- Public invoker: enabled (the portfolio browser must call the API)
- `HF_MODEL_REPO=Nagaraju271/multimodal-image-text-classification-model`
- `ALLOWED_ORIGINS=https://dnagaraju271.github.io`
- `HF_TOKEN`: a Hugging Face fine-grained read token for this private model repository, stored in Google Secret Manager

The Cloud Run free tier is usage-based, not a hard spending cap. A Google Cloud billing account is required. Check the billing page and configure a budget alert before deploying. Do not use a minimum instance.

## 1. Prepare Google Cloud

Create a Google Cloud project in the Cloud Console, link a billing account, and enable the Cloud Run, Cloud Build, Artifact Registry and Secret Manager APIs.

Open Cloud Shell from the Google Cloud Console (the terminal icon in the top-right). It already includes the Google Cloud CLI and is authenticated to your Google account.

Clone this public source repository and select your project:

```bash
git clone https://github.com/dnagaraju271/multimodal-image-text-classification.git
cd multimodal-image-text-classification
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

## 2. Store the Hugging Face token as a secret

Create a fine-grained Hugging Face token that has read access only to the private model repository. Do not put it in source code, a Dockerfile, the frontend, or a committed environment file.

In the terminal, enter the token without echoing it:

```bash
read -s HF_TOKEN
printf '%s' "$HF_TOKEN" | gcloud secrets create hf-model-token --replication-policy=automatic --data-file=-
unset HF_TOKEN
```

If the secret already exists, add a new version instead:

```bash
read -s HF_TOKEN
printf '%s' "$HF_TOKEN" | gcloud secrets versions add hf-model-token --data-file=-
unset HF_TOKEN
```

Grant the Cloud Run runtime service account permission to read the secret. The default Compute Engine service account is shown below; use the service account configured for the service if you choose a different identity:

```bash
PROJECT_ID="$(gcloud config get-value project)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 3. Deploy from the project root

Run from the root of this repository, where the Dockerfile is located:

```bash
gcloud run deploy multimodal-api \
  --source . \
  --region europe-west9 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --concurrency 1 \
  --min 0 \
  --max 1 \
  --timeout 300 \
  --set-env-vars "HF_MODEL_REPO=Nagaraju271/multimodal-image-text-classification-model,ALLOWED_ORIGINS=https://dnagaraju271.github.io" \
  --set-secrets "HF_TOKEN=hf-model-token:latest"
```

The source build uses the root `Dockerfile`. The `.dockerignore` intentionally excludes local datasets, model files and results; the service downloads only its four required model artifacts from the private Hugging Face repository.

## 4. Verify the service

```bash
SERVICE_URL="$(gcloud run services describe multimodal-api --region europe-west9 --format='value(status.url)')"
echo "$SERVICE_URL"
curl -fsS "$SERVICE_URL/api/health"
curl -fsS "$SERVICE_URL/api/models/status"
```

The model status must report `ready: true` and all three models plus the vocabulary available. The first prediction request will load the Keras models into memory and may be slower than subsequent requests.

After the API is verified, set its HTTPS URL in the portfolio page's API connection field and run an actual image + caption prediction.

## Cost and safety notes

- Cloud Run request-based billing can scale the service to zero when idle.
- The maximum instance count is set to one to limit runaway scaling; this also limits concurrent users.
- Free-tier quotas are shared at the billing-account level and are not a spending cap. Build, image storage, network egress, secret usage, or usage beyond free quotas may incur charges.
- Create a billing budget alert before deployment and monitor actual usage.
- Keep the Hugging Face model repository private and rotate the fine-grained token if it is exposed.
