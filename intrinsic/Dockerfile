# The IOC pose estimator service image with its networks running on LiteRT.
#
# Layers the LiteRT pipeline onto Intrinsic's ioc_pose_estimator_image: the
# service's code, configuration and gRPC API stay unchanged, but RF-DETR and
# FoundationPose run in-process with LiteRT (GPU via LiteRT's WebGPU
# accelerator if there is a hardware GPU, else XNNPACK on the CPU) instead of
# being sent to the ML inference service.
#
# Build from the repository root (models are Git LFS files):
#   podman build -f intrinsic/Dockerfile \
#       --build-arg BASE_IMAGE=<ioc_pose_estimator_image> \
#       -t ioc-pose-estimator-litert .
#
# The service image runs Python 3.11 from its Bazel runfiles on distroless
# Debian 13, so the Python module is built for Python 3.11 on Debian 13, with
# a minimal static OpenCV (core, imgproc) so that it needs no extra libraries.

ARG BASE_IMAGE
ARG SERVICE_RUNFILES=/intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator/ioc_service_main.runfiles

FROM docker.io/library/python:3.11-slim-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates libvulkan-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pybind11==3.0.1
# OpenCV 4.10 (as Ubuntu 26.04's): only the modules the pipeline uses.
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz \
    && tar xzf 4.10.0.tar.gz && cmake -S opencv-4.10.0 -B opencv-build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_LIST=core,imgproc \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF \
      -DWITH_OPENMP=OFF -DWITH_PTHREADS_PF=ON -DWITH_EIGEN=OFF \
      -DWITH_LAPACK=OFF -DBUILD_ZLIB=ON -DCMAKE_INSTALL_PREFIX=/opt/opencv \
      -DWITH_TIFF=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_WEBP=OFF \
      -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
      -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_GTK=OFF \
      -DWITH_QT=OFF -DWITH_PROTOBUF=OFF -DBUILD_opencv_python3=OFF \
      -DBUILD_JAVA=OFF -DWITH_ADE=OFF \
    && cmake --build opencv-build -j"$(nproc)" --target install \
    && rm -rf opencv-4.10.0 opencv-build 4.10.0.tar.gz
COPY cpp /src/cpp
COPY python /src/python
RUN cmake -S /src/cpp -B /build -DCMAKE_BUILD_TYPE=Release \
      -DPERCEPTION_BUILD_PYTHON=ON -DPERCEPTION_BUILD_TESTS=OFF \
      -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
      -DPython3_EXECUTABLE="$(which python3.11)" \
    && cmake --build /build -j"$(nproc)" --target litert_pose_estimation \
    && mkdir -p /litert \
    && cp /build/litert_pose_estimation*.so /litert/ \
    && cp /build/litert_lib/ai_edge_litert/libLiteRt.so \
          /build/litert_lib/ai_edge_litert/libLiteRtWebGpuAccelerator.so /litert/ \
    && cp -L /usr/lib/$(gcc -dumpmachine)/libvulkan.so.1 /litert/

# The service's Bazel bootstrap, pointed at litert_service_main.py.
FROM ${BASE_IMAGE} AS base
FROM docker.io/library/python:3.11-slim-trixie AS bootstrap
ARG SERVICE_RUNFILES
COPY --from=base ${SERVICE_RUNFILES}/intrinsic-core+/intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator/_ioc_service_main_stage2_bootstrap.py /bootstrap.py
RUN python3 -c "import re; p='/bootstrap.py'; s=open(p).read(); \
s=re.sub(r'^MAIN_PATH = .*$', 'MAIN_PATH = \"/litert/litert_service_main.py\"', s, count=1, flags=re.M); \
open(p, 'w').write(s)"

FROM ${BASE_IMAGE}
ARG SERVICE_RUNFILES
COPY --from=build /litert/ /litert/
COPY --from=bootstrap /bootstrap.py /litert/bootstrap.py
COPY intrinsic/litert_backend.py intrinsic/litert_service_main.py /litert/
COPY models/rfdetr_seg.tflite models/foundationpose_refine.tflite \
     models/foundationpose_score_b128.tflite models/foundationpose_score_b24.tflite \
     models/foundationpose_score_b240.tflite models/foundationpose_score_b40.tflite \
     /litert/models/
# /litert also holds the Vulkan loader for LiteRT's GPU accelerator; the GPU
# driver's Vulkan ICD comes from the container runtime (e.g. the NVIDIA
# Container Toolkit with NVIDIA_DRIVER_CAPABILITIES=graphics).
ENV RUNFILES_DIR=${SERVICE_RUNFILES} \
    LD_LIBRARY_PATH=/litert \
    LITERT_MODELS_DIR=/litert/models \
    LITERT_ACCELERATOR=auto \
    LITERT_CPU_FALLBACK=0 \
    PYTHONPATH=/litert
ENTRYPOINT ["/intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator/ioc_service_main.runfiles/rules_python++python+python_3_11_aarch64-unknown-linux-gnu/bin/python3", "/litert/bootstrap.py"]
