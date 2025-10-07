NAME="sk080"
IMAGE_NAME="llm-data-analyzer"
VERSION="1.0.0"

# Docker Hub 계정 정보
DOCKER_HUB_USER="progamm3r"
DOCKER_HUB_PASSWORD="dckr_pat_TcN9d9lpV1V-ivAMjhE3DMUAo4U"
DOCKER_CACHE="--no-cache"

# 1. Docker Hub 로그인
echo ${DOCKER_HUB_PASSWORD} | docker login -u ${DOCKER_HUB_USER} --password-stdin \
    || { echo "Docker 로그인 실패"; exit 1; }

# 2. 이미지 빌드
docker build ${DOCKER_CACHE} -t ${NAME}-${IMAGE_NAME}:${VERSION} . \
    || { echo "Docker 이미지 빌드 실패"; exit 1; }

# 3. Docker Hub용 태그 추가
docker tag ${NAME}-${IMAGE_NAME}:${VERSION} ${DOCKER_HUB_USER}/${IMAGE_NAME}:${VERSION}

# 4. 이미지 푸시
docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:${VERSION} \
    || { echo "Docker 이미지 푸시 실패"; exit 1; }

echo "Docker Hub 업로드 완료: ${DOCKER_HUB_USER}/${IMAGE_NAME}:${VERSION}"