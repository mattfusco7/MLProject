FROM python:3.7

ENV LANG C.UTF-8

ENV APP_HOME /app 
WORKDIR $APP_HOME 

RUN python3 --version

COPY . .

RUN ls

RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD python3 main.py