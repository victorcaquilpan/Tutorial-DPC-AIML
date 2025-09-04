#!/bin/bash

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a create_secrets.log
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    log "Error: kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

# Check if sufficient arguments are provided
if [ "$#" -ne 2 ]; then
    log "Usage: $0 ACCESS_TOKEN EMAIL"
    log "Example: $0 glpat-xxxxxxxxxxxx firstname.lastname@aiml.team"
    exit 1
fi

# Assign variables from input arguments
ACCESS_TOKEN=$1
EMAIL=$2
DOCKER_REGISTRY_URL=docker.aiml.team

# Extract username from email
USERNAME="${EMAIL%@*}"

log "Starting secrets creation for user: $USERNAME"

# Check if secrets already exist
if kubectl get secret gitlab-token &> /dev/null; then
    log "Warning: gitlab-token already exists. Delete it first with: kubectl delete secret gitlab-token"
    exit 1
fi

if kubectl get secret gitlab-docker-secret &> /dev/null; then
    log "Warning: gitlab-docker-secret already exists. Delete it first with: kubectl delete secret gitlab-docker-secret"
    exit 1
fi

# Create the general token (gitlab-token)
log "Creating gitlab-token secret..."
kubectl create secret generic gitlab-token \
  --from-literal=access-token=$ACCESS_TOKEN

if [ $? -eq 0 ]; then
    log "✓ gitlab-token created successfully"
else
    log "✗ Failed to create gitlab-token"
    exit 1
fi

# Create the Docker registry secret (gitlab-docker-secret)
log "Creating gitlab-docker-secret..."
kubectl create secret docker-registry gitlab-docker-secret \
  --docker-server=$DOCKER_REGISTRY_URL \
  --docker-username=$USERNAME \
  --docker-password=$ACCESS_TOKEN \
  --docker-email=$EMAIL

if [ $? -eq 0 ]; then
    log "✓ gitlab-docker-secret created successfully"
else
    log "✗ Failed to create gitlab-docker-secret"
    exit 1
fi

# Verify secrets have been created
log "Verifying secrets..."
kubectl get secrets | grep -E "gitlab-token|gitlab-docker-secret"

log "Script execution completed successfully!"
log "Your secrets are now configured in namespace: $(kubectl config view --minify -o jsonpath='{..namespace}')"