FROM python:3.8

RUN pip install sklearn nltk Flask numpy wtforms gunicorn

COPY src/ app/
WORKDIR /app

ENV PORT 8080

CMD exec gunicorns --bind :$PORT --workers 1 --threads 8 app:app
