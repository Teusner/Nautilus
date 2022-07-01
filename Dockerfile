FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update && apt-get -y install git clang-12 wget make --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Installing Cmake
ENV CMAKE_VERSION=3.23
ENV CMAKE_BUILD=2

RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION.$CMAKE_BUILD/cmake-$CMAKE_VERSION.$CMAKE_BUILD-linux-x86_64.sh -O cmake.sh \
    && chmod +x cmake.sh && mkdir -p /opt/cmake && ./cmake.sh --skip-license --prefix=/opt/cmake

ENV PATH="/opt/cmake/bin:${PATH}"

# Installing xtl
RUN git clone https://github.com/xtensor-stack/xtl/ -b master --single-branch && cd xtl \
    && mkdir build && cd build && cmake .. -DCMAKE_C_COMPILER=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-12 && make && make install

# Installing xtensor
RUN git clone https://github.com/xtensor-stack/xtensor/ -b master --single-branch && cd xtensor \
    && mkdir build && cd build && cmake .. -DCMAKE_C_COMPILER=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-12 && make -j8 && make install

# WakeBoat
RUN mkdir -p /Nautilus/build
COPY cuda /Nautilus/cuda
COPY example /Nautilus/example
COPY extern /Nautilus/extern
COPY include /Nautilus/include
COPY src /Nautilus/src
COPY test /Nautilus/test
COPY CMakeLists.txt /Nautilus

ENV CUDAToolkit_ROOT="/usr/local/cuda-11.2"

RUN cd /Nautilus/build && cmake .. -DCMAKE_C_COMPILER=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-12 -DCMAKE_CUDA_COMPILER=/usr/bin/clang++-12 && make -j8

ENTRYPOINT ["/bin/bash"]