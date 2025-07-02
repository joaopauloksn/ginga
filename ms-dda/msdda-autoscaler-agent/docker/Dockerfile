FROM jpksn/msdda-baseimage:0.0.1

COPY . .

RUN pip install -r requirements.txt

USER msdda

CMD ["python3","-u","src/task.py"]
