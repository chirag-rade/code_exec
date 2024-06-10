FROM python:3

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . .

EXPOSE 80

CMD ["python", "main.py"]
