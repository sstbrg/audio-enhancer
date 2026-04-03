FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py enhance.py evaluate.py prepare_dataset.py ./
COPY models/ models/
COPY data/ data/
COPY metrics/ metrics/
COPY utils/ utils/
COPY configs/ configs/

# Default: run training
ENTRYPOINT ["python", "train.py"]
