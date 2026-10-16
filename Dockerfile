FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
USER user

# The Space variables FS_MODELS, FS_ALLOWED_ORIGINS and the secret FS_ACCESS_CODE are set by upload_to_hf.py.
ENV PORT=7860 FS_COLLECT=0 FS_MOTION=/app/motion_jz.pt FS_NO_HTTPS=1 FS_DEVICE=cpu FS_THREADS=2 FS_MODELS='{"Sign Bridge ASL": "Npcalvin/sign-bridge-model"}' HF_HOME=/tmp/hf PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "server.py"]
