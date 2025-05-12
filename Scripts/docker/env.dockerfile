# You can specify the platform here, but it's more flexible to specify during build
# FROM --platform=linux/amd64 ubuntu:latest AS base

FROM ubuntu:latest AS base

# Define build arguments
ARG BUILD_TYPE=Release
ARG LLVM_VERSION=20.1.4

# Add labels for better image management
LABEL maintainer="jwkimrhkgkr@gmail.com"
LABEL version="${LLVM_VERSION}"
LABEL description="LLVM development environment"

# Install cross-compilation dependencies
RUN apt-get update && apt-get install -yq \
    git \
    clang-15 \
    clang++-15 \
    lld \
    cmake \
    ninja-build \
    python3 python3-pip \
    binutils-multiarch

FROM base AS install-llvm

WORKDIR /app
RUN git clone -b llvmorg-${LLVM_VERSION} --depth 1 https://github.com/llvm/llvm-project.git

# Build core components in Release mode
RUN cmake -S llvm-project/llvm -B build-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;lldb" \
    -DLLVM_ENABLE_RUNTIMES="libc;libunwind;libcxxabi;libcxx;compiler-rt;openmp" \
    -DLLVM_INSTALL_UTILS=True && \
    cmake --build build-release --target install

# Build MLIR in Debug or Relase mode
RUN cmake -S llvm-project/llvm -B build-debug -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DLLVM_INSTALL_UTILS=True \
    -DLLVM_ENABLE_PROJECTS="mlir" && \
    cmake --build build-debug --target install

# remove build folders
RUN rm -rf build-release build-debug llvm-project

