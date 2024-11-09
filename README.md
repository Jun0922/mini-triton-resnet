# minikube local image 동기화
```
eval $(minikube -p minikube docker-env)
```

# 이미지 빌드
```
docker build
```

# k8s 실행
## 모델 추출 Job 실행
```
kubectl apply -f ./k8s/job-resnet-training.yaml
```

## triton 서버 실행
```

```

## triton client 실행
```

```