
FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6



COPY best.pt best.pt
COPY app.py app.py
COPY config.py config.py
COPY outputTemplate.py outputTemplate.py  
COPY output_handler.py output_handler.py
COPY forecasting.py forecasting.py
COPY video_processor.py video_processor.py
COPY customCounter.py customCounter.py

CMD ["python", "app.py"]
