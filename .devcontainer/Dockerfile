# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

# (you can now omit user-creation entirely)
# Install extra utilities if you need:
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install debugpy numpy numba scipy pytest black flake8 isort

WORKDIR /workspace