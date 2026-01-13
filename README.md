
<h1>
Project description</h1>

<h3>
The primary objective of this project is to develop and evaluate robust predictive models for electricity price forecasting across distinct power markets. As energy grids increasingly integrate variable renewable energy sources, the volatility of electricity prices has risen, making accurate day-ahead forecasting essential for grid operators and market participants. We aim to model the complex interactions between the exegenous features: electricity generation, the electricity demand load and some autoregressive patterns .
</h3>


<h3>
To ensure the reliability and reproducibility of our results, we will utilize the open access benchmark datasets proposed in the widely cited literature on day-ahead forecasting found in (https://www.sciencedirect.com/science/article/pii/S0306261921004529?via%3Dihub). This source provides high quality, standardized data that facilitates a rigorous comparison of different forecasting methodologies against established metrics. By leveraging these verified benchmarks, we ensure that our performance evaluations are consistent with current academic standards and allow for direct comparison with existing studies in the field.
</h3>

<h3>
Our methodological approach is designed to cover a comprehensive spectrum of predictive techniques, moving from established baselines to experimental deep learning architectures. We will initially employ classical statistical models to capture the fundamental seasonal and linear trends inherent in the time-series data. Building upon this, we will implement XGBoost, a gradient boosting framework that serves as a strong industry standard for regression tasks due to its ability to handle feature interactions effectively.
</h3>
<h3>
Beyond traditional machine learning, we will investigate the efficacy of deep learning approaches. This includes implementing classic neural network architectures to capture high-dimensional non-linear dependencies. Maybe we will include some state-of-the-art models like Mamba and xLSTM. Of course most of the work is going to be in the MLops aspect of the course. 
</h3>

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


<h1>
to start run the following commands
</h1>
<p>
conda create -n <conda name> python=3.13
</p>

<p>
conda activate <conda name>

</p>

<p>
pip install -r requirements.txt
</p>

<p>
run src/data/make_dataset
</p>

<p>
run src/models/train_model
</p>