FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py train.py ./
COPY model.json dv.pkl ./

EXPOSE 9696
ENV PORT=9696

ENTRYPOINT ["python", "predict.py"]

