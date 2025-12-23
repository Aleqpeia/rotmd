# syntax=docker/dockerfile:1.6

FROM mambaorg/micromamba:1.5.8

# Create env with runtime dependencies
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba create -y -n sta -c conda-forge \
    python=3.11 \
    mdanalysis>=2.7 invoke>=2.2 pyyaml>=6.0 fastapi>=0.115 uvicorn>=0.32 \
    jinja2>=3.1 sqlmodel>=0.0.22 email-validator>=2.2 python-multipart>=0.0.17 \
    scikit-learn-extra>=0.3 numpy matplotlib plotly>=5.24 python-kaleido>=0.2 \
    freesasa hole2 dssp \
 && micromamba clean --all --yes

ENV MAMBA_DEFAULT_ENV=sta
ENV PATH=/opt/conda/envs/sta/bin:$PATH
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Install system dependencies for Chrome
USER root
RUN apt-get update && apt-get install -y \
    # Chrome dependencies for Plotly exports
    libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 \
    libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Plotly exports (as root to avoid permission issues)
RUN micromamba run -n sta plotly_get_chrome -y

# Copy source and install the package
WORKDIR /app
COPY . /app
RUN chown -R mambauser:mambauser /app
USER mambauser

# Install the package into the env (as non-root)
RUN micromamba run -n sta python -m pip install --no-cache-dir .

WORKDIR /work
EXPOSE 8000

# Entrypoint: start the server by default
ENTRYPOINT ["bash", "-lc"]
CMD ["sta-server", "--host", "0.0.0.0", "--port", "8000"] 
