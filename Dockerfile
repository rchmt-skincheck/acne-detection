FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
  g++ \
  git \
  graphviz \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED True

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

COPY . ./

COPY best.pt /app/model/best.pt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]