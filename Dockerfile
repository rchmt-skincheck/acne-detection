FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
  g++ \
  git \
  graphviz \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  wget \
  python3-tk && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
  rm /tmp/requirements.txt

# use base from gcr.io/capstone-skincheckai/ml-base
# FROM gcr.io/capstone-skincheckai/ml-base

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

# Download file model machine learning
RUN curl -o best.pt https://storage.googleapis.com/ml-acne-models/best.pt

COPY . ./

# Move the model file to the desired location
RUN mkdir -p /app/model
RUN mv best.pt /app/model/best.pt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# CMD ["python", "app.py"]

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]