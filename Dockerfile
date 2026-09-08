FROM openeuler/openeuler:24.03-lts-sp3

ARG TARGETARCH
ARG HTTP_PROXY=http://127.0.0.1:11082
ARG HTTPS_PROXY=http://127.0.0.1:11082
ARG NO_PROXY=localhost,127.0.0.1

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV no_proxy=${NO_PROXY}

RUN dnf update -y && \
    dnf install -y \
        gcc-toolset-14-* \
        java-21-openjdk-devel \
        wget \
        unzip \
        zip \
        git \
        git-lfs \
        which \
        findutils \
        python \
        python3 \
        python3-pip \
        openssl-devel \
        snappy-devel \
        ca-certificates \
    && dnf clean all

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk
ENV PATH=/opt/openEuler/gcc-toolset-14/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-14/root/usr/lib64:${LD_LIBRARY_PATH}

RUN gcc --version && g++ --version && java -version

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
ARG TENSORFLOW_PIP_VERSION=2.20.0

RUN python3 -m pip config set global.index-url ${PIP_INDEX_URL} && \
    python3 -m pip config set global.trusted-host ${PIP_TRUSTED_HOST}

RUN python3 -m pip install --no-cache-dir --upgrade setuptools wheel && \
    python3 -m pip install --no-cache-dir --prefer-binary \
        atomgit \
        tensorflow==${TENSORFLOW_PIP_VERSION}

RUN python3 -c "import tensorflow as tf; print(tf.__version__)"

ARG BAZEL_VERSION=7.4.1

RUN case "${TARGETARCH:-$(uname -m)}" in \
        amd64|x86_64) bazel_arch=x86_64 ;; \
        arm64|aarch64) bazel_arch=arm64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH:-$(uname -m)}" >&2; exit 1 ;; \
    esac && \
    wget https://mirrors.huaweicloud.com/bazel/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${bazel_arch} \
        -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

RUN bazel --version


WORKDIR /workspace/tensorflow
COPY . .

ARG TF_SYSTEM_LIBS=boringssl,snappy
ARG BAZEL_DISTDIR=/workspace/tensorflow/distdir

RUN case "${TARGETARCH:-$(uname -m)}" in \
        amd64|x86_64) bazel_cpu_copt="--copt=-march=native" ;; \
        arm64|aarch64) bazel_cpu_copt="--copt=-march=armv8.5-a" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH:-$(uname -m)}" >&2; exit 1 ;; \
    esac && \
    mkdir -p ${BAZEL_DISTDIR} && \
    bazel --output_user_root=./output build \
        --noenable_bzlmod \
        --experimental_repo_remote_exec \
        --copt=-O3 \
        --host_copt=-O3 \
        --cxxopt=-std=c++17 \
        --host_cxxopt=-std=c++17 \
        ${bazel_cpu_copt} \
        --host_copt=-I/usr/local/include \
        --linkopt=-L/usr/local/lib64 \
        --host_linkopt=-L/usr/local/lib64 \
        --linkopt=-Wl,-rpath,/usr/local/lib64 \
        --host_linkopt=-Wl,-rpath,/usr/local/lib64 \
        --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} \
        --action_env=TF_SYSTEM_LIBS=${TF_SYSTEM_LIBS} \
        --repo_env=HTTP_PROXY=${HTTP_PROXY} \
        --repo_env=HTTPS_PROXY=${HTTPS_PROXY} \
        --repo_env=http_proxy=${HTTP_PROXY} \
        --repo_env=https_proxy=${HTTPS_PROXY} \
        --repo_env=NO_PROXY=${NO_PROXY} \
        --repo_env=no_proxy=${NO_PROXY} \
        --distdir=${BAZEL_DISTDIR} \
        --check_direct_dependencies=off \
        //:predictor_server //:brpc_client && \
    cp -L bazel-bin/predictor_server /usr/local/bin/predictor_server && \
    cp -L bazel-bin/brpc_client /usr/local/bin/brpc_client

ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV http_proxy=
ENV https_proxy=
ENV NO_PROXY=
ENV no_proxy=
