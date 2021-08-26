PROJECT_ID="lazarus-testing-dc6d0"
SERVICE_NAME="matt-model-testing"


cd src && gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}