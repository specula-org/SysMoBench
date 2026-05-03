# SysMoBench: full benchmark image with TLA+ tooling, all host build chains,
# and the claude-code CLI used by transition validation and the agent
# invariant translator.
#
# Build:    docker build -t sysmobench .
# Run:      docker run --rm -it sysmobench bash
# With API: docker run --rm -it -e ANTHROPIC_API_KEY=... sysmobench bash
#
# Asterinas-based tasks (spin, mutex, rwmutex, ringbuffer) launch their
# own containers — pass `-v /var/run/docker.sock:/var/run/docker.sock`
# to let those harnesses talk to the host Docker daemon.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        openjdk-21-jdk-headless \
        maven \
        golang-go \
        git curl ca-certificates build-essential gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 from NodeSource (Ubuntu 24.04's apt npm bundles Node 18, which
# is below the minimum the claude-code CLI now requires).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --no-fund --no-audit @anthropic-ai/claude-code@2.1.126

WORKDIR /opt/sysmobench
COPY . /opt/sysmobench

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install -e . \
    && /opt/venv/bin/sysmobench-setup

ENV PATH="/opt/venv/bin:${PATH}"

CMD ["bash"]
