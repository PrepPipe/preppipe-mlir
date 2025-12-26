FROM ubuntu:25.10

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && \
    apt-get install -y software-properties-common git build-essential ninja-build cmake clang-20 clang-format-20 llvm-20-dev libmlir-20-dev mlir-20-tools && \
    apt-get install -y python3-full python3-pip && \
    apt-get install unzip curl wget && \
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

# antlr4-runtime
# the runtime zip file does not contain LICENSE.txt which makes installation fail, so we create an empty one
RUN mkdir -p antlr4/runtime/antlr4-runtime && cd antlr4/runtime/antlr4-runtime && \
    wget https://www.antlr.org/download/antlr4-cpp-runtime-4.13.2-source.zip && \
    unzip antlr4-cpp-runtime-4.13.2-source.zip && \
    echo "" > ../../LICENSE.txt && \
    mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja install && \
    cd ../../.. && rm -rf antlr4

CMD ["bash"]
