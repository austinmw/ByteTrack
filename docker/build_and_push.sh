ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

ACCOUNT_ID=197237853439
REGION=us-west-2
REPO_NAME=bytetrack/training

# Log in to private ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Create ECR repository if it doesn't already exist
aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

# Build docker image
nvidia-docker build -f Dockerfile -t $REPO_NAME .

# Tag it
docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Push it
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
