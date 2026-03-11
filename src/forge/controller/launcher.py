# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (SLURM, Kubernetes, etc.)"""

import atexit
import logging
import re

from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch.actor import ProcMesh
from monarch.job import JobState, JobTrait, SlurmJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"
_DEFAULT_MONARCH_PORT = 26600

# Monarch's KubernetesJob.add_mesh() requires mesh names to be
# lowercase alphanumeric only (RFC 1123 + Monarch hostname restriction).
# TorchForge uses underscores (e.g. "generator_0", "replay_buffer"), so
# we strip non-alphanumeric characters before registering meshes.
_MESH_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9]")


def _sanitize_mesh_name(name: str) -> str:
    """Strip non-alphanumeric characters for K8s mesh name compliance.

    Monarch's KubernetesJob.add_mesh() validates: lowercase, alphanumeric only,
    starts with alpha, max 63 chars. This converts names like ``generator_0``
    to ``generator0`` and ``replay_buffer`` to ``replaybuffer``.
    """
    sanitized = _MESH_NAME_SANITIZE_RE.sub("", name.lower())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "m" + sanitized  # ensure starts with alpha
    return sanitized[:63]


class _AliasedJobState:
    """Wrapper around JobState that maps original mesh names to sanitized ones.

    The provisioner looks up HostMeshes via ``getattr(job_state, mesh_name)``
    using the original names from config (e.g. ``generator_0``). But
    KubernetesJob registers them under sanitized names (e.g. ``generator0``).
    This wrapper transparently resolves the mapping.
    """

    def __init__(self, job_state: JobState, name_map: dict[str, str]):
        self._job_state = job_state
        self._name_map = name_map  # original -> sanitized

    def __getattr__(self, name: str):
        # Check if this is an original name that needs mapping
        sanitized = self._name_map.get(name)
        if sanitized is not None:
            return getattr(self._job_state, sanitized)
        # Fall through to the real JobState for anything else
        return getattr(self._job_state, name)


def get_meshes_from_config(cfg: LauncherConfig) -> dict[str, int]:
    """Extract mesh requirements from launcher config.

    Args:
        cfg: The launcher configuration

    Returns:
        Dictionary mapping mesh names to number of hosts required
    """
    meshes: dict[str, int] = {}

    # Add services that need remote hosts
    # Expand services with multiple replicas into per-replica meshes
    for service_name, service_cfg in cfg.services.items():
        hosts = getattr(service_cfg, "hosts", None)
        if hosts and hosts > 0:
            base_mesh_name = service_cfg.mesh_name or service_name
            num_replicas = service_cfg.num_replicas
            for replica_idx in range(num_replicas):
                mesh_name = f"{base_mesh_name}_{replica_idx}"
                meshes[mesh_name] = hosts

    # Add actors that need remote hosts
    for actor_name, actor_cfg in cfg.actors.items():
        hosts = getattr(actor_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = actor_cfg.mesh_name or actor_name
            meshes[mesh_name] = hosts

    return meshes


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a single SlurmJob for all resources.

        This pre-allocates all meshes defined in the config in one Slurm job.

        Returns:
            A tuple of (job, job_state) containing the SlurmJob handle and its state.
        """
        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Build slurm_args from config
        slurm_args = [f"--{key}={value}" for key, value in self.cfg.slurm_args.items()]

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        job = SlurmJob(
            meshes=meshes,  # e.g., {"generator_0": 1, "generator_1": 1, "trainer": 2}
            slurm_args=slurm_args,
            job_name=self.cfg.job_name + "_workers" or "forge_job",
            time_limit="72:00:00",  # Default to 72 hours
            gpus_per_node=self.cfg.gpus_per_node,
            cpus_per_task=self.cfg.cpus_per_task,
            mem=self.cfg.mem,
        )

        # Apply the job to allocate resources
        logger.info("Submitting SlurmJob...")
        job.apply()
        logger.info("SlurmJob submitted, waiting for allocation...")

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for job allocation
        logger.info("Getting job state (this will block until nodes are allocated)...")
        job_state = job.state(cached_path=None)

        logger.info("SlurmLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


class K8sLauncher(BaseLauncher):
    """Kubernetes launcher implementation.

    A thin wrapper over Monarch's KubernetesJob that translates TorchForge's
    LauncherConfig into KubernetesJob mesh specifications. The MonarchMesh
    operator handles the actual pod provisioning (StatefulSets, headless
    Services, etc.).

    For each mesh, the launcher builds a V1PodSpec from the configured
    container image (via Monarch's ImageSpec helper) and passes it to
    KubernetesJob.add_mesh(). The pod spec is patched with additional
    configuration such as a memory-backed /dev/shm volume when shm_size
    is set.

    Per-mesh overrides can be specified in k8s_args using the mesh name
    as key. Supported override keys:
      - image: Container image (overrides k8s_image)
      - resources: Resource requests/limits (e.g. {"nvidia.com/gpu": 1})
      - labels: Additional K8s labels for the mesh pods
      - shm_size: Per-mesh /dev/shm size (overrides global shm_size)

    Top-level k8s_args keys:
      - shm_size: Global /dev/shm size applied to all meshes
      - rdma_enabled: Set to true if the cluster supports RDMA networking

    Args:
        cfg: The launcher configuration containing K8s-specific fields.
    """

    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a KubernetesJob for all resources.

        This provisions MonarchMesh CRDs for all meshes defined in the config.
        The monarch-kubernetes operator reconciles these CRDs into StatefulSets
        and headless Services, then the KubernetesJob waits for pods to be ready.

        Returns:
            A tuple of (job, job_state) containing the KubernetesJob handle
            and its state with allocated HostMesh objects.

        Raises:
            ValueError: If k8s_image is not set and no per-mesh image is provided.
            ImportError: If monarch.job.kubernetes is not available (requires
                torchmonarch >= 0.3.0).
        """
        try:
            from kubernetes import client as k8s_client
            from monarch.job.kubernetes import ImageSpec, KubernetesJob
        except ImportError as err:
            raise ImportError(
                "KubernetesJob requires torchmonarch >= 0.3.0 and the "
                "kubernetes client. Install with: "
                "pip install 'torchmonarch>=0.3.0' kubernetes"
            ) from err

        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Validate that we have an image to use
        if not self.cfg.k8s_image and not self.cfg.k8s_args:
            raise ValueError(
                "k8s_image must be set in the launcher config, or per-mesh "
                "images must be provided in k8s_args. Example:\n"
                "  provisioner:\n"
                "    launcher: k8s\n"
                '    k8s_image: "your-registry/torchforge-worker:latest"'
            )

        # Build KubernetesJob kwargs
        k8s_kwargs = {"namespace": self.cfg.k8s_namespace}
        if self.cfg.k8s_timeout is not None:
            k8s_kwargs["timeout"] = self.cfg.k8s_timeout

        logger.info(f"Creating KubernetesJob with meshes: {meshes}")
        job = KubernetesJob(**k8s_kwargs)

        # Sanitize mesh names for K8s compliance and build a mapping so the
        # provisioner can still look up HostMeshes by their original names.
        name_map: dict[str, str] = {}  # original -> sanitized

        # Global shm_size from k8s_args applies to all meshes unless
        # overridden per-mesh.
        global_shm_size = self.cfg.k8s_args.get("shm_size", None)

        # Add each mesh to the job
        for mesh_name, num_hosts in meshes.items():
            sanitized_name = _sanitize_mesh_name(mesh_name)
            name_map[mesh_name] = sanitized_name

            mesh_overrides = self.cfg.k8s_args.get(mesh_name, {})
            mesh_image = mesh_overrides.get("image", self.cfg.k8s_image)
            mesh_labels = mesh_overrides.get("labels", None)
            mesh_shm_size = mesh_overrides.get("shm_size", global_shm_size)

            # Convert resources to a plain dict — OmegaConf DictConfig
            # objects break the kubernetes client which accesses attributes
            # like .openapi_types that don't exist as config keys.
            mesh_resources = mesh_overrides.get("resources", None)
            if mesh_resources is not None:
                mesh_resources = {str(k): str(v) for k, v in mesh_resources.items()}

            if not mesh_image:
                raise ValueError(
                    f"No container image specified for mesh '{mesh_name}'. "
                    f"Set k8s_image in the launcher config or provide a "
                    f"per-mesh image in k8s_args.{mesh_name}.image"
                )

            # Build the base V1PodSpec from ImageSpec using Monarch's
            # built-in helper, then customize it (e.g. /dev/shm volume)
            # before passing it as pod_spec to add_mesh().
            image_spec = ImageSpec(mesh_image, resources=mesh_resources)
            pod_spec = KubernetesJob._build_worker_pod_spec(
                image_spec, _DEFAULT_MONARCH_PORT
            )

            # Patch in a memory-backed /dev/shm when shm_size is set.
            # K8s defaults /dev/shm to 64 MiB which is too small for
            # PyTorch shared-memory operations (model state dicts, etc.).
            if mesh_shm_size:
                shm_volume = k8s_client.V1Volume(
                    name="dshm",
                    empty_dir=k8s_client.V1EmptyDirVolumeSource(
                        medium="Memory",
                        size_limit=str(mesh_shm_size),
                    ),
                )
                shm_mount = k8s_client.V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm",
                )
                pod_spec.volumes = (pod_spec.volumes or []) + [shm_volume]
                pod_spec.containers[0].volume_mounts = (
                    pod_spec.containers[0].volume_mounts or []
                ) + [shm_mount]

            add_mesh_kwargs = {
                "name": sanitized_name,
                "num_replicas": num_hosts,
                "pod_spec": pod_spec,
            }

            if mesh_labels:
                add_mesh_kwargs["labels"] = mesh_labels

            logger.info(
                f"Adding mesh '{mesh_name}' -> '{sanitized_name}' "
                f"with image '{mesh_image}'"
                + (f", shm_size={mesh_shm_size}" if mesh_shm_size else "")
                + f" ({num_hosts} replicas)"
            )

            job.add_mesh(**add_mesh_kwargs)

        # Apply the job to create MonarchMesh CRDs
        logger.info("Creating MonarchMesh CRDs via KubernetesJob...")
        job.apply()
        logger.info("MonarchMesh CRDs created, waiting for pods to be ready...")

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for pod allocation
        logger.info(
            "Getting job state (this will block until pods are running)..."
        )
        job_state = job.state(cached_path=None)

        # Wrap the JobState so the provisioner can look up HostMeshes by
        # original config names (e.g. "generator_0") even though they were
        # registered under sanitized names (e.g. "generator0").
        aliased_state = _AliasedJobState(job_state, name_map)

        logger.info("K8sLauncher initialization complete.")
        return job, aliased_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.K8S:
        return K8sLauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
