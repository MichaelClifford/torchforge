# Running experiments on Kubernetes


## Prerequisites and Setup


1. A Kubernetes cluster with the [monarch-kubernetes operator](https://github.com/meta-pytorch/monarch-kubernetes) installed:
   ```
   helm repo add monarch-operator https://meta-pytorch.github.io/monarch-kubernetes
   helm repo update
   helm install monarch-operator monarch-operator/monarch-operator \
     --namespace monarch-system --create-namespace
   ```

2. A worker container image with Monarch and TorchForge installed.
   Build from the provided Dockerfile:
   ```
   docker build -t torchforge-worker:latest -f docker/Dockerfile .
   ```
   Push to your cluster's image registry as needed.

3. A namespace for the job with appropriate RBAC. The submit script `submit.sh`
   creates the namespace and RBAC automatically if not present.

## To run GRPO training:

The controller runs inside a pod on the cluster. The submit script handles
namespace creation, RBAC setup, and launching the controller pod.

```
./experimental/k8s/submit.sh qwen3-1b
```

The controller pod provisions MonarchMesh CRDs for all GPU and CPU actors.
The monarch-kubernetes operator reconciles these into StatefulSets and
headless Services. Once all worker pods are running, the training loop begins.

## Configuration

The K8s launcher is configured via the `provisioner:` section in the YAML config:

```yaml
provisioner:
  launcher: k8s
  job_name: my-job
  k8s_namespace: torchforge       # Kubernetes namespace
  k8s_image: "my-image:latest"    # Default worker image
  k8s_timeout: 600                # Pod readiness timeout (seconds)
  k8s_args:                       # Optional per-mesh overrides
    generator:
      image: "custom-image:latest"
      labels:
        team: ml-infra
```

The `services:` and `actors:` top-level sections define resource allocations,
same as for SLURM. Actors with `hosts: 1` (or higher) are provisioned as
remote pods; actors without `hosts` run locally on the controller.

## Monitoring your run

List all pods in the namespace:
```bash
kubectl get pods -n torchforge -o wide
```

Stream logs from each component:
```bash
# Controller (orchestrates the training loop)
kubectl logs -f forge-controller -n torchforge

# Trainer worker pod
kubectl logs -f $(kubectl get pods -n torchforge -l monarch.pytorch.org/mesh-name=trainer -o jsonpath='{.items[0].metadata.name}') -n torchforge

# Generator worker pod
kubectl logs -f $(kubectl get pods -n torchforge -l monarch.pytorch.org/mesh-name=generator0 -o jsonpath='{.items[0].metadata.name}') -n torchforge
```

Check GPU utilization on a worker pod:
```bash
kubectl exec -it <pod-name> -n torchforge -- nvidia-smi
```

View MonarchMesh CRD status:
```bash
kubectl get monarchmeshes -n torchforge
```

