FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
# Install necessary system packages and Python 3.10
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip

# Copy requirements.txt first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -U flash-attn --no-build-isolation \
    && pip install --no-cache-dir -r requirements.txt

# Copy all other application files
COPY . .

RUN chmod +x entrypoint.sh

# Expose the Streamlit port (default is 8501)
EXPOSE 8501

ENTRYPOINT ["entrypoint.sh"]