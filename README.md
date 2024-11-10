# prerequisite
```
- docker engine 이 미리 설치돼 있어야 합니다.
- docker 가상화 용량이 모자라 image pull이 안될 수 있습니다.
docker desktop 설정에서 resources 탭에 virtual disk limit 을 150g 정도로 올릴 것을 권장드립니다.
```

# minikube 설치
```
# https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download
# 해당 url 에 접속 후 본인 OS 에 맞는 minikube 를 설치
minikube start

# minikube local image 동기화
minikube docker-env
# 위 명령어 실행 후 나오는 맨 아랫 줄의 주석 없이 복사 후 커맨드 실행 (아래는 mac OS 기준)
eval $(minikube -p minikube docker-env)

# terminal 하나 추가로 열어서 dashboard 열기
minikube dashboard
```

# 이미지 빌드
```
# project root 로 이동 (~/mini-triton-resnet)
docker build -t resnet-training -f ./docker/Dockerfile.resnet_training .
docker build -t mini-triton-server -f ./docker/Dockerfile.triton_server .
docker build -t triton-client -f ./docker/Dockerfile.triton_client .
```

# pv / pvc 배포
```
# project root 로 이동 (~/mini-triton-resnet)
kubectl apply -f ./k8s/pv-data-storage.yaml
kubectl apply -f ./k8s/pvc-data-storage.yaml
kubectl apply -f ./k8s/pv-model-storage.yaml
kubectl apply -f ./k8s/pvc-model-storage.yaml
```

# service 배포
```
# project root 로 이동 (~/mini-triton-resnet)
kubectl apply -f ./k8s/service-triton-server.yaml
```

# 과제 배포
## 1. ResNet Training Job
```
# k8s/job-resnet-training.yaml 안의 TRAIN_EPOCHS 를 원하는 숫자로 변경 (default 5)
# project root 로 이동 (~/mini-triton-resnet)
kubectl apply -f ./k8s/job-resnet-training.yaml

# job 완수 대기 이후 학습 완료된 모델 ckpt 파일 확인 후 경로 복사
minikube ssh
cd /mnt/data/model-storage/ConvNets/ResNet/lightning_logs/version_0/checkpoints
# minikube ssh 종료
exit
```


## 2. triton 서버 실행
```
# k8s/deployment-triton-server.yaml 을 열어서 
# env의 BEST_MODEL_PATH 에 위에서 확인한 경로를 ConvNets/~ 부터 ckpt 파일까지 기입
# project root 로 이동 (~/mini-triton-resnet)
kubectl apply -f ./k8s/deployment-triton-server.yaml
# minikube dashboard 에서 해당 pod 가 배포되어 정상적으로 로그가 찍히는지 확인
```


## 3. triton client 실행
```
# project root 로 이동 (~/mini-triton-resnet)
# triton_server 가 정상적으로 기동한 후에 배포해야 오류나지 않음
kubectl apply -f ./k8s/pod-triton-client.yaml
# minikube dashboard 에서 해당 pod 의 로그 확인
```