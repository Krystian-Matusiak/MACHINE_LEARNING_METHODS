FROM python:3.10

ARG UID
ARG GID

WORKDIR /app

RUN groupadd --gid $GID user1 && useradd --create-home -u $UID --gid $GID user1 && chown -R user1 /app

COPY requirements.txt .
ENV PYTHONPATH="/app/"

RUN pip --no-cache-dir install -r requirements.txt

USER user1
