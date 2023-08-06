# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
#COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install kafka-python==2.0.2 psycopg2-binary==2.9.7 python-dotenv==1.0.0

COPY src/ /usr/local/lib/python3.9/site-packages
# Define the command to run your script
WORKDIR /app/src

###CMD ["ls", "/usr/local/lib/python3.9/site-packages"]
CMD ["python", "scripts/kc.py"]
