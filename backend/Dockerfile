FROM python:3.11-buster

RUN pip install poetry

COPY CONFIG.py .
COPY main_API.py .
COPY main_BACK.py .
COPY poetry.lock .
COPY pyproject.toml .

RUN poetry install

EXPOSE 8000