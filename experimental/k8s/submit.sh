#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Submit a TorchForge GRPO training job to Kubernetes.
#
# This script:
#   1. Creates the namespace and RBAC if they don't exist
#   2. Creates a ConfigMap from the local YAML config file
#   3. Launches a controller pod that mounts and uses the config
#   4. The controller runs the standard GRPO entrypoint (apps.grpo.main)
#   5. The K8sLauncher provisions MonarchMesh CRDs for worker pods
#
# The config is injected via a ConfigMap, so changes to the local YAML
# are picked up without rebuilding the worker image.
#
# Usage:
#   ./experimental/k8s/submit.sh qwen3-1b
#
# Prerequisites:
#   - kubectl configured to access a K8s cluster
#   - monarch-kubernetes operator installed
#   - Worker image available (see docker/Dockerfile)

set -euo pipefail

CONFIG_NAME="${1:?Usage: $0 <config_name>  (e.g., qwen3-1b)}"
CONFIG_PATH="experimental/k8s/${CONFIG_NAME}.yaml"

NAMESPACE="torchforge"
CONTROLLER_IMAGE="${CONTROLLER_IMAGE:-torchforge-worker:latest}"
TORCHSTORE_RDMA_ENABLED="${TORCHSTORE_RDMA_ENABLED:-0}"
CONFIGMAP_NAME="forge-config-$(echo "${CONFIG_NAME}" | tr '_' '-')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ ! -f "${REPO_ROOT}/${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${REPO_ROOT}/${CONFIG_PATH}"
    echo "Available configs:"
    ls "${SCRIPT_DIR}"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml
    exit 1
fi

echo ">>> Submitting K8s job: ${CONFIG_NAME}"
echo "    Config: ${CONFIG_PATH}"
echo "    Namespace: ${NAMESPACE}"
echo "    Controller image: ${CONTROLLER_IMAGE}"
echo

# ---- Create namespace and RBAC ----
echo ">>> Creating namespace and RBAC..."
kubectl create namespace "${NAMESPACE}" 2>/dev/null || true

kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: forge-controller
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: forge-controller
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["monarch.pytorch.org"]
    resources: ["monarchmeshes"]
    verbs: ["create", "get", "list", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: forge-controller
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: forge-controller
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: forge-controller
  apiGroup: rbac.authorization.k8s.io
EOF

# ---- Create ConfigMap from local config file ----
echo ">>> Creating ConfigMap '${CONFIGMAP_NAME}' from ${CONFIG_PATH}..."
kubectl delete configmap "${CONFIGMAP_NAME}" -n "${NAMESPACE}" --ignore-not-found 2>/dev/null
kubectl create configmap "${CONFIGMAP_NAME}" \
    --from-file=config.yaml="${REPO_ROOT}/${CONFIG_PATH}" \
    -n "${NAMESPACE}"

# ---- Launch controller pod ----
echo ">>> Launching controller pod..."

# Delete existing controller pod if present
kubectl delete pod forge-controller -n "${NAMESPACE}" --ignore-not-found --wait=true 2>/dev/null

kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: forge-controller
  namespace: ${NAMESPACE}
  labels:
    app: forge-controller
    config: ${CONFIG_NAME}
spec:
  serviceAccountName: forge-controller
  volumes:
    - name: config-volume
      configMap:
        name: ${CONFIGMAP_NAME}
  containers:
    - name: controller
      image: ${CONTROLLER_IMAGE}
      imagePullPolicy: IfNotPresent
      command:
        - python
        - -m
        - apps.grpo.main
        - --config
        - /etc/forge/config.yaml
      env:
        - name: TORCHSTORE_RDMA_ENABLED
          value: "${TORCHSTORE_RDMA_ENABLED}"
      workingDir: /workspace/torchforge
      volumeMounts:
        - name: config-volume
          mountPath: /etc/forge
          readOnly: true
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
  restartPolicy: Never
EOF

echo ">>> Controller pod submitted."
echo
echo "Monitor with:"
echo "  kubectl logs -f forge-controller -n ${NAMESPACE}"
echo "  kubectl get pods -n ${NAMESPACE}"
echo
echo "Cleanup:"
echo "  kubectl delete pod forge-controller -n ${NAMESPACE}"
echo "  kubectl delete configmap ${CONFIGMAP_NAME} -n ${NAMESPACE}"
echo "  kubectl delete namespace ${NAMESPACE}"
