FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p app/model

COPY . .

RUN ls -la app/model/ || echo "Model klasörü boş"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]