FROM python:3.10.7

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]