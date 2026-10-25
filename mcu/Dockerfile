FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc-arm-none-eabi libnewlib-arm-none-eabi binutils-arm-none-eabi qemu-system-arm make python3 && rm -rf /var/lib/apt/lists/*
