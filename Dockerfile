FROM python:3

ADD . /diabetes-python
WORKDIR /diabetes-python

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]

