# Ginga: Adaptive Microservices Autoscaling Framework

**Ginga** is an open‑source implementation of the **MS‑DDA architecture** — a data‑driven and adaptive solution for autoscaling microservices in Kubernetes environments. It was developed as part of the master’s dissertation *“An Adaptive Data‑Driven Architecture for Microservices Autoscaling”* by João Paulo Karol Santos Nunes (USP, 2025).

> 🧠 Ginga combines real‑time monitoring, machine‑learning predictions, and infrastructure awareness to scale microservices efficiently — reducing cloud costs and improving performance.

---

## 📌 Highlights

- ✅ Real‑time ML‑based autoscaling (horizontal **and** vertical)
- 📊 Prometheus + MongoDB metrics backend
- ⚙️ Custom Kubernetes Operator with CRD
- ☁️ Fully containerised for OpenShift/Kubernetes
- 🧪 Open‑source tool born from academic research

---

## 📦 Architecture

Ginga follows the **MS‑DDA** architecture, whose key building blocks are:

| Component | Purpose |
|-----------|---------|
| **Custom Resource Definition (CRD)** | Declares microservice‑specific autoscaling parameters |
| **Deployment Operator** | Watches CRDs and triggers the predict‑and‑scale loop |
| **Model Predictor** | Loads a trained neural network (`.h5`) to predict required replicas |
| **Scaler Service** | Applies replica changes via the Kubernetes API |
| **Monitoring Layer** | Prometheus collects CPU, memory, throughput and latency metrics |
| **Data Store** | MongoDB stores historical metrics for training and inference |

> A detailed diagram is available in `docs/ginga_architecture.png` or Figure 7 of the dissertation.

---

## 🚀 Getting Started

> **Prerequisites:** A Kubernetes/OpenShift cluster with Prometheus **and** MongoDB reachable by the pods.

### 1. Clone the repository

```bash
git clone https://github.com/joaopauloksn/ginga.git
cd ginga
```

### 2. Build & push the Docker image (example)

```bash
docker build -t <your-registry>/ginga-predictor:latest -f docker/Dockerfile .
docker push <your-registry>/ginga-predictor:latest
```

### 3. Deploy Ginga components

```bash
kubectl apply -f k8s/crds/
kubectl apply -f k8s/operator/
```

### 4. Provide the ML model

Ginga expects a trained model at:

```text
/shared-volume/model.h5
```

Copy your `best_model.h5` there (or mount a volume that contains it).

### 5. Deploy a sample microservice

```bash
kubectl apply -f samples/iotoccupancy-deployment.yaml
```

---

## 🧠 Machine‑Learning Model

The bundled neural network is trained on real‑world operational metrics and labelled using service‑level objectives (SLOs).

- **Inputs:** CPU, memory, delivered/undelivered messages, latency, etc.  
- **Output:** Optimal number of replicas (e.g., 1, 2 or 3)

To retrain your own model, see the companion repo **msdda‑model‑development**.

---

## 📁 Project Layout

```
ginga/
├── docker/                 # Dockerfile & requirements
├── k8s/                    # CRDs and operator manifests
├── samples/                # Example workloads
├── src/
│   ├── task.py
│   └── model_deployment/
│       ├── predict.py
│       └── scaler.py
├── models/                 # best_model.h5 (not committed)
└── README.md
```

---

## 📊 Evaluation Summary

In an IoT case study, Ginga outperformed Kubernetes HPA:

* **CPU usage:** ↓ 50 %  
* **Memory usage:** ↓ 87 %  
* **Replica count:** ↓ 90 % (thanks to vertical + horizontal scaling)

See Chapter 4 of the dissertation for full experiment details.

---

## 📚 Citation

> Nunes, J. P. K. S. (2025). *An Adaptive Data‑Driven Architecture for Microservices Autoscaling*. Master’s Dissertation – ICMC/USP.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🙌 Acknowledgements

- Instituto de Ciências Matemáticas e de Computação – USP  
- Prof. Elisa Yumi Nakagawa for supervision and guidance  
