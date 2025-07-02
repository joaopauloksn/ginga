FROM jpksn/msdda-baseimage:0.0.1

# Copy only requirements.txt first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

USER msdda

CMD ["python3","-u","src/task.py"]
