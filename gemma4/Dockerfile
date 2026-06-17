FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DLLAMA_AVX=ON -DLLAMA_AVX2=ON"
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]