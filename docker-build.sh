NAME="sk080"
IMAGE_NAME="llm-data-analyzer"
CPU_PLATFORM=amd64
VERSION="1.0.0"

# Docker 이미지 빌드
docker build \
  --tag ${NAME}-${IMAGE_NAME}:${VERSION} \
  --file Dockerfile \
  --platform linux/${CPU_PLATFORM} \
  ${IS_CACHE} .
