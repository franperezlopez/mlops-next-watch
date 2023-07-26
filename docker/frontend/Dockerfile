FROM python:3.9


WORKDIR /app

RUN pip3 install --no-cache-dir fastapi==0.100.0 streamlit==1.25.0 uvicorn==0.23.1

EXPOSE 8000
EXPOSE 8501

WORKDIR /app/src/frontend

#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "&", "streamlit", "run", "app.py", "--server.port", "8501"]
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 --reload & streamlit run app.py --server.port 8501"]
