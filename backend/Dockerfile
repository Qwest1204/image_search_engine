FROM python:3.11-bookworm

RUN pip install poetry

COPY CONFIG.py .
COPY main_API.py .
COPY utils.py .
COPY poetry.lock .
COPY pyproject.toml .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN poetry install