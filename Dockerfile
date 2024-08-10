FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages and Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to set python3 and pip to point to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.10 1

# Check Python versions
RUN python3 --version
RUN pip --version

# Copy requirements.txt first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy all other application files
COPY . .

# Expose the Streamlit port (default is 8501)
EXPOSE 8501