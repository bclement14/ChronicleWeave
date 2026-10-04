# Pinned description of the working image `chronicleweave-whisperx` (built 2025-04-29).
# docker/requirements-lock.txt was produced with `pip freeze` inside that image.
# NOT build-tested: this file documents the image; a rebuild is a separate decision.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/bin/python && python -m pip install --upgrade pip

COPY docker/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r /tmp/requirements-lock.txt

# Models are not baked in: the pipeline mounts ~/.cache/chronicleweave at /root/.cache.
WORKDIR /app
CMD ["whisperx", "--help"]
