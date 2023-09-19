<!-- Next Watch
==============================

MLOps  MLOps project for movie recommendations. -->

<p align="center">
<img src="images/nextwatch.png" alt="Logo" width="210" height="110"/>
</p>

# Next Watch: E2E MLOps Pipelines with Spark!
![CI](https://github.com/brnaguiar/mlops-next-watch/actions/workflows/behavioral_tests.yml/badge.svg?event=push)


## Prerequisites
- Python
- Conda or Venv
- Docker

## Installation and Quick Start
1. Clone the repo
```sh
git clone https://github.com/brnaguiar/mlops-next-watch.git

```

2. Create environment
```sh
make env
```

3. Activate conda env
```sh
source activate nwenv
```

4. Install requirements / dependencies and assets
```sh
make dependencies
```

5. Pull the datasets
```sh
make datasets
```

6. Configure containers and secrets
```sh
make init
```

7. Run Docker Compose
```sh
make run
```

8. Populate production Database with users
```sh
make users
```

## Useful Service Endpoints
```
- Jupyter `http://localhost:8888`
- MLFlow `http://localhost:5000`
- Minio Console `http://localhost:9001`
- Airflow `http://localhost:8080`
- Streamlit Frontend `http://localhost:8501`
- FastAPI Backend` http://localhost:8000/`
- Grafana Dashboard `http://localhost:3000`
- Prometheus `http://localhost:9090`
- Pushgateway `http://localhost:9091`
- Spark UI `http://localhost:8081`
```

## Architecture
<img src="./images/project_diagram.jpg"/>

## Service Endpoints Showcase

### Streamlit Frotend App
<img src="./images/streamlit_ui.png"/>

### MLflow UI
<img src="./images/mlflow_ui.png"/>

### Minio UI
<img src="./images/minio_ui.png"/>

### Airflow UI
<img src="./images/airflow_ui.png"/>

### Grafana UI
<img src="./images/grafana_ui.png"/>

### Prometheus UI
<img src="./images/prometheus_ui.png"/>

### Prometheus Drift Detection Example
<img src="./images/prometheus_ui_drift_warning.png"/>

<!-- #4. Create a `.env` file (`.env` sample below)#
5. Run the the project!
```sh
make run
```
-->

<!-- PROJECT LOGO -->
<!--
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
-->


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 

<!-- #cookiecutterdatascience</small></p> -->
