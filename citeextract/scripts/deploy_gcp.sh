#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "[error] line $LINENO: command \"$BASH_COMMAND\" exited $?" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
  echo "[preflight] loaded $PROJECT_ROOT/.env"
fi

: "${PROJECT_ID:?required — pick a neutral name; avoid your real name for double-blind review}"
: "${BILLING_ACCOUNT_ID:?required — list with: gcloud billing accounts list}"
: "${OPENAI_API_KEY:?required — the project-scoped key from DEPLOY.md §2}"
: "${REVIEW_ACCESS_TOKEN:?required — generate with: python3 -c 'import secrets; print(secrets.token_urlsafe(16))'}"

if [[ ${#REVIEW_ACCESS_TOKEN} -lt 16 ]]; then
  echo "[preflight] REVIEW_ACCESS_TOKEN looks too short (${#REVIEW_ACCESS_TOKEN} chars: '${REVIEW_ACCESS_TOKEN}')." >&2
  echo "            This is almost always a fragmented shell expansion. Set the value" >&2
  echo "            in $PROJECT_ROOT/.env or paste the literal value, then re-run." >&2
  exit 1
fi

REGION="${REGION:-us-central1}"
BUDGET_USD="${BUDGET_USD:-200}"
GROBID_IMAGE="${GROBID_IMAGE:-grobid/grobid:0.8.2-crf}"
GROBID_MIN_INSTANCES="${GROBID_MIN_INSTANCES:-1}"  # 1 = always warm (~€23/mo), 0 = scale to zero (~€5/mo, but reviewers eat ~30s cold starts)
MAIN_CPU_ALWAYS="${MAIN_CPU_ALWAYS:-0}"
SA_NAME="checkcitation-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

command -v gcloud >/dev/null 2>&1 \
  || { echo "[preflight] gcloud not on PATH — install from https://cloud.google.com/sdk/docs/install" >&2; exit 1; }

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
  echo "[preflight] no active gcloud account — run: gcloud auth login" >&2
  exit 1
fi

if [[ ! "$BILLING_ACCOUNT_ID" =~ ^[0-9A-Fa-f]{6}-[0-9A-Fa-f]{6}-[0-9A-Fa-f]{6}$ ]]; then
  echo "[preflight] BILLING_ACCOUNT_ID '$BILLING_ACCOUNT_ID' is not in XXXXXX-XXXXXX-XXXXXX form" >&2
  echo "[preflight] list valid IDs with: gcloud billing accounts list" >&2
  exit 1
fi

[[ "$OPENAI_API_KEY" == sk-* ]] \
  || { echo "[preflight] OPENAI_API_KEY does not start with 'sk-' — wrong key?" >&2; exit 1; }

cd "$(dirname "$0")/.."

if [[ ! -f .gcloudignore ]]; then
  cat > .gcloudignore <<'EOF'
#!include:.gitignore
.git/
.gcloudignore
data/
paper/
experiment/
baselines/
tests/
*.pdf
*.pre_recovery
EOF
  echo "[setup] wrote .gcloudignore"
fi

if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud projects create "$PROJECT_ID" --name="checkcitation"
fi
gcloud config set project "$PROJECT_ID" --quiet
gcloud config set run/region "$REGION" --quiet
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"

gcloud services enable billingbudgets.googleapis.com cloudbilling.googleapis.com --quiet

BUDGET_NAME="checkcitation-trial"
BILLING_CURRENCY=$(gcloud billing accounts describe "$BILLING_ACCOUNT_ID" \
  --format="value(currencyCode)" 2>/dev/null)
BILLING_CURRENCY="${BILLING_CURRENCY:-USD}"
existing_budget=$(gcloud billing budgets list \
  --billing-account="$BILLING_ACCOUNT_ID" \
  --filter="displayName=${BUDGET_NAME}" \
  --format="value(name)" 2>/dev/null | head -n1 || true)

if [[ -z "$existing_budget" ]]; then
  gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT_ID" \
    --display-name="$BUDGET_NAME" \
    --budget-amount="${BUDGET_USD}${BILLING_CURRENCY}" \
    --threshold-rule=percent=0.25 \
    --threshold-rule=percent=0.50 \
    --threshold-rule=percent=0.90 \
    --threshold-rule=percent=1.0 \
    --filter-projects="projects/${PROJECT_ID}"
else
  gcloud billing budgets update "$existing_budget" \
    --billing-account="$BILLING_ACCOUNT_ID" \
    --budget-amount="${BUDGET_USD}${BILLING_CURRENCY}" >/dev/null
fi

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --display-name="checkcitation runtime"
fi

upsert_secret() {
  local name="$1" value="$2"
  [[ -n "$value" ]] || { echo "[error] secret '$name' has empty value" >&2; return 1; }
  if gcloud secrets describe "$name" >/dev/null 2>&1; then
    printf "%s" "$value" | gcloud secrets versions add "$name" --data-file=- >/dev/null
  else
    printf "%s" "$value" | gcloud secrets create "$name" \
      --replication-policy=automatic --data-file=- >/dev/null
  fi
  gcloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role=roles/secretmanager.secretAccessor >/dev/null 2>&1 || true
}
upsert_secret openai-api-key       "$OPENAI_API_KEY"
upsert_secret review-access-token  "$REVIEW_ACCESS_TOKEN"

SECRETS_FLAG="OPENAI_API_KEY=openai-api-key:latest,REVIEW_ACCESS_TOKEN=review-access-token:latest"
if [[ -n "${S2_API_KEY:-}" ]]; then
  upsert_secret s2-api-key "$S2_API_KEY"
  SECRETS_FLAG="${SECRETS_FLAG},S2_API_KEY=s2-api-key:latest"
fi
if [[ -n "${SERPAPI_KEY:-}" ]]; then
  upsert_secret serpapi-key "$SERPAPI_KEY"
  SECRETS_FLAG="${SECRETS_FLAG},SERPAPI_KEY=serpapi-key:latest"
fi

gcloud run deploy grobid \
  --image="$GROBID_IMAGE" \
  --region="$REGION" \
  --port=8070 \
  --memory=4Gi --cpu=2 \
  --min-instances="$GROBID_MIN_INSTANCES" --max-instances=2 \
  --allow-unauthenticated \
  --timeout=300 \
  --startup-probe=httpGet.path=/api/isalive,httpGet.port=8070,initialDelaySeconds=10,periodSeconds=10,failureThreshold=20,timeoutSeconds=5 \
  --quiet

GROBID_URL=$(gcloud run services describe grobid --region="$REGION" --format='value(status.url)')
echo "[deploy] GROBID at $GROBID_URL"

REPO="cloud-run-source-deploy"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/checkcitation:latest"
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" --quiet
fi

build_start=$(date +%s)
gcloud builds submit --tag="$IMAGE" --timeout=1800s
echo "[build] image build took $(( $(date +%s) - build_start ))s"

cpu_flag=""
[[ "$MAIN_CPU_ALWAYS" == "1" ]] && cpu_flag="--no-cpu-throttling"

ENV_VARS="GROBID_SERVICE_URL=${GROBID_URL},DAILY_ANALYSIS_LIMIT=10,HOURLY_IP_LIMIT=3,MONTHLY_BUDGET_USD=150"
[[ -n "${CROSSREF_MAILTO:-}" ]] && ENV_VARS="${ENV_VARS},CROSSREF_MAILTO=${CROSSREF_MAILTO}"
[[ -n "${OPENALEX_MAILTO:-}" ]] && ENV_VARS="${ENV_VARS},OPENALEX_MAILTO=${OPENALEX_MAILTO}"

gcloud run deploy checkcitation \
  --image="$IMAGE" \
  --region="$REGION" \
  --port=7860 \
  --memory=4Gi --cpu=1 \
  --min-instances=1 --max-instances=1 \
  --concurrency=5 \
  $cpu_flag \
  --service-account="$SA_EMAIL" \
  --set-env-vars="$ENV_VARS" \
  --set-secrets="${SECRETS_FLAG}" \
  --allow-unauthenticated \
  --timeout=900 \
  --startup-probe=httpGet.path=/health,httpGet.port=7860,initialDelaySeconds=15,periodSeconds=10,failureThreshold=20,timeoutSeconds=5 \
  --quiet

URL=$(gcloud run services describe checkcitation --region="$REGION" --format='value(status.url)')

echo "[smoke] GET ${URL}/health"
http_code=$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 30 "${URL}/health" || echo "000")
if [[ "$http_code" != "200" ]]; then
  echo "[smoke] FAILED — /health returned HTTP $http_code" >&2
  echo "[smoke] inspect logs: gcloud run services logs read checkcitation --region=${REGION} --limit=50" >&2
  exit 1
fi
echo "[smoke] /health returned 200"

cat <<EOF

[done]
  URL:           ${URL}
  Token:         ${REVIEW_ACCESS_TOKEN}
  Reviewer link: ${URL}/?token=<token-above>
  Logs:          gcloud run services logs read checkcitation --region=${REGION}

To rotate the token after camera-ready:
  printf "%s" "<new-token>" | gcloud secrets versions add review-access-token --data-file=-
  gcloud run services update checkcitation --region=${REGION} \\
    --update-secrets="REVIEW_ACCESS_TOKEN=review-access-token:latest"
EOF
