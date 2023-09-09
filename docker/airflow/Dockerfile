FROM apache/airflow:2.7.0-python3.9

WORKDIR /app

COPY requirements.minimal .

USER root
RUN apt update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
ENV PATH="${JAVA_HOME}/bin:${PATH}"
RUN export JAVA_HOME

USER airflow
RUN pip install apache-airflow-providers-apache-spark==4.1.4
RUN pip install -r requirements.minimal
#RUN pip install -e .
