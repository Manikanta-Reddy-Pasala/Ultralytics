## ------------------------------- Builder Stage ------------------------------ ##
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR=/app/python

WORKDIR /app

COPY ./.python-version .
COPY ./pyproject.toml .

RUN uv sync

## ------------------------------- Production Stage ------------------------------ ##
FROM ubuntu:22.04 AS production

RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/python /app/python

RUN mkdir -p /app/SAMPLES_LOW_POWER

COPY 2G_MODEL/best_int8_openvino_model /app/2G_MODEL/best_int8_openvino_model
COPY 3G_4G_MODEL/best.pt /app/3G_4G_MODEL/best.pt
COPY dummy.jpg /app
COPY ./*.py /app

ENV PATH="/app/.venv/bin:$PATH"

# Memory tuning
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV MALLOC_ARENA_MAX=2
ENV MAX_THREADS=2
ENV TORCH_THREADS=4

EXPOSE 4444

CMD ["python3", "scanner.py"]
