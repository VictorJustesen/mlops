
# Project description

The primary objective of this project is to develop and evaluate robust predictive models for electricity price forecasting across distinct power markets. As energy grids increasingly integrate variable renewable energy sources, the volatility of electricity prices has risen, making accurate day-ahead forecasting essential for grid operators and market participants. We aim to model the complex interactions between the exegenous features: electricity generation, the electricity demand load and some autoregressive patterns .

To ensure the reliability and reproducibility of our results, we will utilize the open access benchmark datasets proposed in the widely cited literature on day-ahead forecasting found in (https://www.sciencedirect.com/science/article/pii/S0306261921004529?via%3Dihub). This source provides high quality, standardized data that facilitates a rigorous comparison of different forecasting methodologies against established metrics. By leveraging these verified benchmarks, we ensure that our performance evaluations are consistent with current academic standards and allow for direct comparison with existing studies in the field.

Our methodological approach is designed to cover a comprehensive spectrum of predictive techniques, moving from established baselines to experimental deep learning architectures. We will initially employ classical statistical models to capture the fundamental seasonal and linear trends inherent in the time-series data. Building upon this, we will implement XGBoost, a gradient boosting framework that serves as a strong industry standard for regression tasks due to its ability to handle feature interactions effectively.

Beyond traditional machine learning, we will investigate the efficacy of deep learning approaches. This includes implementing classic neural network architectures to capture high-dimensional non-linear dependencies. Maybe we will include some state-of-the-art models like Mamba and xLSTM. Of course most of the work is going to be in the MLops aspect of the course.

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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## To start run the following commands (or see Makefile for more options):

1. **Create and activate environment:**
   ```bash
   make create_environment
   source .venv/bin/activate  # or conda activate mlops
   ```

2. **Install dependencies:**
   ```bash
   make requirements
   ```

3. **Process data:**
   ```bash
   make data
   ```

**Note, this repository is made, and we recommend using UV, if already installed, just;**
   ```bash
   uv sync
   ```

in which additional commands to download the data can be run such as;

   ```bash
   uv run make_data
   uv run rnn_data
   uv run train_rnn
   ```

**Additional note**

As we tested base functions, we found out the online download of the data from Zenodo can be disrupted if the service is offline, or if the network is 'flagged as suspicious connection'. Therefore the raw data has been supplied for this project.

## Workflows

### Local Workflow

Local development cycle on your machine:

```bash
# 1. Build images (native architecture)
make build-local

# 2. Train model
make train-local

# 3. Deploy API locally
make deploy-local
```

The API will be available at `http://localhost:8080`

### Cloud Workflow

Cloud deployment on GCP:

```bash
# 1. Build and push images
make build-cloud

# 2. Train model on Vertex AI
make train-cloud

# 3. Deploy API to Cloud Run
make deploy-cloud

# 4. Get API URL
make get-api-url
```

Download trained model from cloud to local:
```bash
make download-model
```


# GCP

## Initial Setup

1. **Authenticate with GCP**
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>
```

2. **Configure Docker for GCP Artifact Registry**
```bash
gcloud auth configure-docker gcr.io
```

3. **Create GCP bucket**
```bash
gcloud storage buckets create gs://mlops-dataset-84636 \
  --location=europe-west4 \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access

gcloud storage buckets update gs://mlops-dataset-84636 --versioning
```

4. **Initialize DVC with GCP remote**
```bash
make dvc-init
```

5. **Set up email alerts for Cloud Run errors (OPTIONAL)**

**Easiest method:** Use GCP Console
- Go to: **Monitoring > Alerting > Create Policy**
- Select **Cloud Run Revision** resource
- Metric: `Request count` filtered by `response_code_class = 5xx`
- Threshold: > 0 for 5 minutes
- Add your email in notification channels


**Pull data** (download from cloud to local)
```bash
make dvc-pull
```

**Push data** (upload from local to cloud)
```bash
make dvc-push
```

## Adding New Data Files

```bash
dvc add data/raw/newfile.csv
make dvc-push
```
