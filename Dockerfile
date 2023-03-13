FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libopencv-dev python3-opencv

RUN pip install --upgrade pip

RUN pip install --no-cache-dir tensorflow \
    mtcnn \
    sanic[ext] 
    
RUN pip install --no-cache-dir -r requirements.txt

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

COPY . .

EXPOSE 8000
CMD [ "python", "server.py"]