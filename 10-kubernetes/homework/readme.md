 Homework 10 - Kubernetes 

## Answers

| Question | Answer |
|----------|--------|
| Question 1 | 0.49 |
| Question 2 | kind version 0.20.0 |
| Question 3 | Pod |
| Question 4 | ClusterIP |
| Question 5 | kind load docker-image |
| Question 6 | 9696 |
| Question 7 | subscription |

## Setup

Clone the repository and navigate to homework folder:
git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp/cohorts/2025/05-deployment/homework

Build the Docker image:
docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .

## Question 1: Test Model Locally

Run the Docker container:
docker run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10

text

Test the model:
python q6_test.py

**Answer: 0.49**

## Question 2: Check kind Version

kind --version

**Answer: kind version 0.20.0**

## Question 3: Smallest Deployable Unit

**Answer: Pod**

## Question 4: Default Service Type

Create Kubernetes cluster:
kind create cluster
kubectl get services

**Answer: ClusterIP**

## Question 5: Register Docker Image

kind load docker-image zoomcamp-model:3.13.10-hw10

**Answer: kind load docker-image**

## Question 6: Container Port

Create `deployment.yaml`:
apiVersion: apps/v1
kind: Deployment
metadata:
name: subscription
spec:
selector:
matchLabels:
app: subscription
replicas: 1
template:
metadata:
labels:
app: subscription
spec:
containers:
- name: subscription
image: zoomcamp-model:3.13.10-hw10
resources:
requests:
memory: "64Mi"
cpu: "100m"
limits:
memory: "128Mi"
cpu: "500m"
ports:
- containerPort: 9696

Apply deployment:
kubectl apply -f deployment.yaml
kubectl get pods

**Answer: 9696**

## Question 7: Service Selector

Create `service.yaml`:
apiVersion: v1
kind: Service
metadata:
name: subscription-service
spec:
type: LoadBalancer
selector:
app: subscription
ports:

port: 80
targetPort: 9696

Apply service:
kubectl apply -f service.yaml
kubectl get services


**Answer: subscription**

## Test the Service

Port forward to test locally:
kubectl port-forward service/subscription-service 9696:80


Run test in another terminal:
python q6_test.py

