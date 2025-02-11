FROM python:3.10-slim
ENV TZ="Europe/Istanbul"

WORKDIR /app

ADD . /app/

# Install required packages
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run" , "Ana_Sayfa.py", "--server.port", "8000", "--server.address", "0.0.0.0"]