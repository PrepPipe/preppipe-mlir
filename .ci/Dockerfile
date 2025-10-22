FROM ubuntu:25.10

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && \
    apt-get install -y software-properties-common git build-essential ninja-build cmake clang-20 clang-format-20 llvm-20-dev libmlir-20-dev mlir-20-tools && \
    apt-get install -y python3-full python3-pip && \
    apt-get install unzip && \
    apt-get clean

# Install these if eventually we still decide to build Python bindings
# python3-dev pybind11-dev python3-pybind11 nanobind-dev python3-nanobind

# Install lit testing tool (LLVM somehow does not come with it)
RUN pip install lit --break-system-packages --root-user-action=ignore

# create symlinks without version suffix for convenience
RUN ln -s /usr/bin/clang-20 /usr/bin/clang && \
    ln -s /usr/bin/clang++-20 /usr/bin/clang++ && \
    ln -s /usr/bin/clang-format-20 /usr/bin/clang-format

ENV CMAKE_PREFIX_PATH=/usr/lib/llvm-20/lib/cmake/mlir
CMD ["bash"]
