# prerequisite
```
1. docker 가상화 용량이 모자라 image pull이 안될 수 있습니다.
docker desktop 설정에서 resources 탭에 virtual disk limit 을 150g 정도로 올릴 것을 권장드립니다.
```

# minikube local image 동기화
```
eval $(minikube -p minikube docker-env)
```

# pv / pvc 배포
```
kubectl apply -f ./k8s/pv-data-storage.yaml
kubectl apply -f ./k8s/pvc-data-storage.yaml
kubectl apply -f ./k8s/pv-model-storage.yaml
kubectl apply -f ./k8s/pvc-model-storage.yaml
```

# 이미지 빌드
```
docker build
```

# k8s 실행
## 모델 추출 Job 실행
```
kubectl apply -f ./k8s/job-resnet-training.yaml

# job 완수 대기 이후 학습 완료된 모델 ckpt 파일 확인
minikube ssh
cd /mnt/data/model-storage/ConvNets/ResNet/lightning_logs/version_0/checkpoints
# minikube ssh 종료
exit
```

## triton 서버 실행
```
# k8s/deployment-triton-server.yaml 을 열어서 
# env의 BEST_MODEL_PATH 에 위에서 확인한 경로를 ConvNets/~ 부터 ckpt 파일까지 기입
kubectl apply -f ./k8s/deployment-triton-server.yaml
```

## triton client 실행
```

```