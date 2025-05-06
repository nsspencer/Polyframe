# Use the full Python image (not slim) for the latest Python release
FROM python:3.12

# Install build tools, dependencies, wget, and unzip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install debugpy numpy numba && \
    pip install --upgrade pip setuptools wheel

# Set working directory for your project
WORKDIR /app