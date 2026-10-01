"""The IOC pose estimator service, with its networks running on LiteRT.

Identical to intrinsic-core's ioc_service_main.py (runtime context, config,
gRPC server), except that litert_backend.install() replaces the service's
calls to the ML inference service with the LiteRT pipeline. Environment:

  LITERT_MODELS_DIR   directory with the .tflite models (default /litert/models)
  LITERT_ACCELERATOR  auto (GPU if there is a hardware GPU, else CPU), gpu or
                      cpu (default auto)
  LITERT_GPU_FP16     1 to run on the GPU in float16 (default 0)

Runs with the service's own Python environment, e.g. through the service's
Bazel stage-2 bootstrap with MAIN_PATH pointing to this file (see Dockerfile).
"""

from concurrent import futures
import logging
import os
import pathlib
import sys

from google.protobuf import text_format
import grpc
from intrinsic.perception.proto.v1 import pose_estimation_service_pb2_grpc
from intrinsic.resources.proto import runtime_context_pb2
from intrinsic_perception.intrinsic.perception.service.ioc_pose_estimator.proto import ioc_service_config_pb2
from intrinsic_perception.intrinsic.perception.service.ioc_pose_estimator.service import ioc_pose_estimator_service
from intrinsic_perception.intrinsic.perception.service.ioc_pose_estimator.service import pose_estimator_model
from intrinsic_perception.intrinsic.perception.service.ioc_pose_estimator.service import segmentation_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litert_backend  # pylint: disable=g-import-not-at-top

_RUNTIME_CONTEXT_FILE = "/etc/intrinsic/runtime_config.pb"
_DEFAULT_PORT = 50051
_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", -1),
    ("grpc.max_send_message_length", -1),
    ("grpc.max_message_length", -1),
]


def main():
  logging.basicConfig(level=logging.INFO)
  config = ioc_service_config_pb2.IocPoseEstimatorServiceConfig()
  runtime_context = None
  if pathlib.Path(_RUNTIME_CONTEXT_FILE).exists():
    with open(_RUNTIME_CONTEXT_FILE, "rb") as f:
      runtime_context = runtime_context_pb2.RuntimeContext.FromString(f.read())
    if runtime_context.HasField("config") and runtime_context.config.Is(
        ioc_service_config_pb2.IocPoseEstimatorServiceConfig.DESCRIPTOR):
      runtime_context.config.Unpack(config)
  logging.info("Service config: %s", text_format.MessageToString(config))
  # The service requires its two models as dependencies served by the ML
  # inference service, but LiteRT runs them in-process (install() below
  # replaces them). Without them, e.g. outside a solution, the service is
  # constructed with placeholders.
  for dependency in ("segmentation_model", "foundationpose_model"):
    if not config.HasField(dependency):
      getattr(config, dependency).SetInParent()
  if not config.HasField("inference_service"):
    logging.info("No ML inference service; LiteRT runs the models.")
    segmentation_model.SegmentationModel = lambda **unused: None
    pose_estimator_model.PoseEstimationModel = lambda **unused: None

  servicer = ioc_pose_estimator_service.IocPoseEstimatorService(config=config)
  info = litert_backend.install(
      servicer,
      models_dir=os.environ.get("LITERT_MODELS_DIR", "/litert/models"),
      accelerator=os.environ.get("LITERT_ACCELERATOR", "auto"),
      gpu_fp16=os.environ.get("LITERT_GPU_FP16", "0") == "1")
  logging.info("LiteRT backend installed: %s", info)

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                       options=_GRPC_OPTIONS)
  pose_estimation_service_pb2_grpc.add_PoseEstimationServiceServicer_to_server(
      servicer, server)
  port = (runtime_context.port
          if runtime_context is not None and runtime_context.port > 0
          else _DEFAULT_PORT)
  if server.add_insecure_port(f"[::]:{port}") != port:
    raise RuntimeError(f"Failed to use port {port}")
  server.start()
  logging.info("LiteRT IOC pose estimator listening on port %d", port)
  server.wait_for_termination()


if __name__ == "__main__":
  main()
