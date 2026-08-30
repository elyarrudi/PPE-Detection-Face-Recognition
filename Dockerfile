FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/root/.insightface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --break-system-packages -r requirements.txt

RUN python3 -m pip install --break-system-packages \
        nvidia-cublas-cu12 \
        nvidia-cuda-runtime-cu12 \
    && CUDA_PYTHON_LIB_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia \
    && find "$CUDA_PYTHON_LIB_DIR" -type f -name "*.so*" -exec ln -sf {} /usr/local/lib/ \; \
    && ldconfig

RUN mkdir -p /app/input_video /app/output_video

COPY . .

ENTRYPOINT ["python3", "/app/main_pt.py"]