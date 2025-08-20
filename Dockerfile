FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get install -y --no-install-recommends \
		gcc \
        fonts-noto-cjk \
        fonts-wqy-zenhei \
        fonts-wqy-microhei \
		&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir matplotlib numpy

WORKDIR /work

CMD ["python", "-u", "main.py"]