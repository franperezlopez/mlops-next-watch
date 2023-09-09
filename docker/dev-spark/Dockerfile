# - stage 0 -
FROM ubuntu:20.04 
ARG DEBIAN_FRONTEND=noninteractive

# Install Java 11 and PySpark dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk bash iputils-ping python3.9 python3.9-venv python3.9-dev python3-pip git

# Set environment variables for Java
# Set the JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set the PATH environment variable to include Java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY ./requirements.minimal .
RUN pip3 install --upgrade pip
RUN pip3 install "Cython<3.0" "pyyaml<6" --no-build-isolation

RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN python3.9 -m pip install -r requirements.minimal
ENV PYTHONPATH="/usr/local/lib/python3.9/site-packages:$PYTHONPATH"
#ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:$PYTHONPATH"

WORKDIR /app

# Generate a default Jupyter config file
RUN jupyter notebook --generate-config

# Set the default Jupyter config options
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.Spawner.env_keep.append('PYTHONPATH')" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
EXPOSE 4040-4050

# Set the default command to run Jupyter notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
