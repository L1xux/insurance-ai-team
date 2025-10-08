pipeline {
  agent any

  environment {
    GIT_URL                = 'https://github.com/L1xux/llm-data-analyzer.git'
    GIT_BRANCH             = 'main'           
    GIT_ID                 = 'skala-github-id'   
    IMAGE_NAME             = 'llm-data-analyzer'    
    
    IMAGE_TAG              = "1.0.0-${BUILD_NUMBER}"
    IMAGE_REGISTRY_URL     = 'docker.io'      
    IMAGE_REGISTRY_PROJECT = 'progamm3r'      

    DOCKER_CREDENTIAL_ID   = 'progamm3r'
    K8S_NAMESPACE          = 'skala-practice'
  }

  options {
    disableConcurrentBuilds()
    timestamps()
  }

  stages {
    stage('Clone Repository') {
      steps {
        echo 'Clone Repository'
        git branch: "${GIT_BRANCH}", url: "${GIT_URL}", credentialsId: "${GIT_ID}"
        sh 'ls -al'
      }
    }

    stage('Compute Image Meta') {
      steps {
        script {
          def hashcode = sh(script: "date +%s%N | sha256sum | cut -c1-12", returnStdout: true).trim()
          env.FINAL_IMAGE_TAG = "${IMAGE_TAG}-${hashcode}"
          env.IMAGE_REGISTRY  = "${env.IMAGE_REGISTRY_URL}/${env.IMAGE_REGISTRY_PROJECT}"
          env.REG_HOST        = env.IMAGE_REGISTRY_URL
          env.IMAGE_REF       = "${env.IMAGE_REGISTRY}/${IMAGE_NAME}:${env.FINAL_IMAGE_TAG}"

          echo "REG_HOST: ${env.REG_HOST}"
          echo "IMAGE_REF: ${env.IMAGE_REF}"
        }
      }
    }

    stage('Image Build & Push (docker)') {
      steps {
        script {
          docker.withRegistry('https://index.docker.io/v1/', "${DOCKER_CREDENTIAL_ID}") {
            sh "docker build --platform=linux/amd64 -t ${IMAGE_REGISTRY_PROJECT}/${IMAGE_NAME}:${FINAL_IMAGE_TAG} ."
            sh "docker push ${IMAGE_REGISTRY_PROJECT}/${IMAGE_NAME}:${FINAL_IMAGE_TAG}"
          }
        }
      }
    }

    // Git에 이미지 태그 커밋
    stage('Update Git Repository') {
      steps {
        script {
          withCredentials([usernamePassword(credentialsId: "${GIT_ID}", 
                                           usernameVariable: 'GIT_USER', 
                                           passwordVariable: 'GIT_TOKEN')]) {
            sh '''
            set -eux
            
            # 이미지 태그 변경
            sed -Ei "s#(image:[[:space:]]*)(docker\\.io/)?(${IMAGE_REGISTRY_PROJECT}/${IMAGE_NAME}):[^[:space:]]+#\\1\\2\\3:${FINAL_IMAGE_TAG}#g" ./k8s/deploy.yaml
            
            echo "=== Updated deploy.yaml ==="
            grep "image:" ./k8s/deploy.yaml
            
            # Git 설정
            git config user.email "mr938363@google.com"
            git config user.name "Jin"
            
            # 변경사항 커밋
            git add ./k8s/deploy.yaml
            git commit -m "chore: Update image to ${FINAL_IMAGE_TAG} [skip ci]" || echo "No changes to commit"
            
            # Git 푸시
            git push https://${GIT_USER}:${GIT_TOKEN}@github.com/L1xux/llm-data-analyzer.git HEAD:main
            '''
          }
        }
      }
    }
  }
}