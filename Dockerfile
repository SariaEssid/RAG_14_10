# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Authors
LABEL Author="seddik bouzayani <seddik.bouzayani@esprit.tn>"
LABEL maintainer="seddik bouzayani <seddik.bouzayani@esprit.tn>"

# Set environment variables to optimize Python and container behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/esprit/.local/bin:$PATH"

# Install system dependencies and Python build dependencies (Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -g 2003 esprit && useradd -r -u 2003 -g esprit -m esprit

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements to leverage Docker caching
COPY --chown=esprit:esprit requirements.txt .

# Install Python dependencies as non-root user (to /home/esprit/.local)
USER esprit
RUN pip install --user -r requirements.txt

# Copy the application source code to the container
COPY --chown=esprit:esprit . .

# ──────────────────────────────────────────────────────────────────
# Rebuild the FAISS index inside the container so that index.pkl is
# always serialised with the exact pydantic / langchain versions
# that are installed above.  This prevents the
#   KeyError: '__fields_set__'
# crash that occurs when the pickle was created with a different
# pydantic version than the one running the application.
# ──────────────────────────────────────────────────────────────────
ENV KNOWLEDGE_BASE_FOLDER=/app/app/knowledge_base
RUN python -m app.academic_rag.build_rag_index

# Expose the application port
EXPOSE 2112

# Define the command to run your application
CMD ["python", "main.py"]
