#!/bin/bash

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a docker_login.log
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log "Error: Docker is not installed. Please install Docker and try again."
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

# Check if already logged in
if docker info 2>/dev/null | grep -q "Username: $USERNAME"; then
    log "Already logged in as $USERNAME"
    exit 0
fi

# Perform docker login
log "Attempting to log in to Docker registry: $DOCKER_REGISTRY_URL"
log "Username: $USERNAME"

if echo "$ACCESS_TOKEN" | docker login $DOCKER_REGISTRY_URL --username $USERNAME --password-stdin; then
    log "✓ Docker login successful"
    log "You can now push/pull images to/from $DOCKER_REGISTRY_URL"
else
    log "✗ Docker login failed"
    log "Please check your access token and try again"
    exit 1
fi

log "Script execution completed successfully!"