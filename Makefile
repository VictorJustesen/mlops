.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 test test_coverage pre-commit-install pre-commit-run pre-commit-update dvc-init dvc-pull dvc-push dvc-status

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = mlops
PYTHON_INTERPRETER = python3
GCP_BUCKET ?= mlops-dataset-84636
PROJECT_ID ?= dtumlops-484812
REGION ?= europe-west1
TAG ?= latest
IMAGE_API = gcr.io/$(PROJECT_ID)/rnn-api
IMAGE_TRAINER = gcr.io/$(PROJECT_ID)/rnn-trainer

# Platform detection - use native for local, amd64 for cloud
LOCAL_PLATFORM := $(shell uname -m | sed 's/x86_64/linux\/amd64/' | sed 's/arm64/linux\/arm64/' | sed 's/aarch64/linux\/arm64/')
CLOUD_PLATFORM := linux/amd64

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	@if command -v uv > /dev/null; then \
		uv sync; \
	else \
		$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel; \
		$(PYTHON_INTERPRETER) -m pip install -r requirements.txt; \
	fi

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed
	$(PYTHON_INTERPRETER) src/features/rnn_data.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Initialize DVC and configure GCP remote
dvc-init:
	@echo "Initializing DVC..."
	dvc init
	@echo "Adding GCP remote..."
	dvc remote add -d gcp-remote gs://$(GCP_BUCKET)
	@echo "DVC initialized with GCP remote"

## Pull data from DVC remote (GCP bucket)
dvc-pull:
	@echo "Pulling data from GCP bucket..."
	dvc pull

## Push data to DVC remote (GCP bucket)
dvc-push:
	@echo "Pushing data to GCP bucket..."
	dvc push

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

test:
	pytest tests/ -v

test_coverage:
	pytest tests/ -v --cov=src --cov-report=html

## Format code with ruff
format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

## Lint code with ruff
lint_all:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

check: pre-commit-run lint_all test_coverage

## Install pre-commit hooks
pre-commit-install:
	uv run pre-commit install

## Run pre-commit on all files
pre-commit-run:
	uv run pre-commit run --all-files

## Update pre-commit hooks
pre-commit-update:
	uv run pre-commit autoupdate

fix:
	uv run ruff format .
	uv run ruff check --fix .
#uv run pre-commit run --all-files


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# LOCAL WORKFLOW - Build, Train, Deploy Locally                               #
#################################################################################

## Build images for local development
build-local:
	docker build -t $(IMAGE_TRAINER):local --target trainer --platform $(LOCAL_PLATFORM) .
	docker build -t $(IMAGE_API):local --target api --platform $(LOCAL_PLATFORM) .

## Train model locally in container (saves to ./models/)
train-local: build-local
	docker run --rm \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/data:/app/data \
		$(IMAGE_TRAINER):local

## Deploy API locally (uses local models/ folder)
deploy-local: build-local
	docker run --rm -p 8080:8080 \
		-v $(PWD)/models:/app/models \
		-e MODEL_SOURCE=local \
		$(IMAGE_API):local

#################################################################################
# CLOUD WORKFLOW - Build, Train, Deploy on GCP                                #
#################################################################################

## Build images for cloud deployment and push to GCR
build-cloud:
	docker build -t $(IMAGE_TRAINER):$(TAG) --target trainer --platform $(CLOUD_PLATFORM) .
	docker build -t $(IMAGE_API):$(TAG) --target api --platform $(CLOUD_PLATFORM) .
	docker tag $(IMAGE_TRAINER):$(TAG) $(IMAGE_TRAINER):latest
	docker tag $(IMAGE_API):$(TAG) $(IMAGE_API):latest
	docker push $(IMAGE_TRAINER):$(TAG)
	docker push $(IMAGE_TRAINER):latest
	docker push $(IMAGE_API):$(TAG)
	docker push $(IMAGE_API):latest

## Train model on Vertex AI
train-cloud: build-cloud
	gsutil -m rsync -r data/grouped gs://$(GCP_BUCKET)/data/grouped
	gcloud ai custom-jobs create \
		--region=$(REGION) \
		--display-name=rnn-train-$(shell date +%Y%m%d-%H%M%S) \
		--config=vertex_config.yaml

## Deploy API to Cloud Run
deploy-cloud:
	gcloud run deploy rnn-api \
		--image=$(IMAGE_API):$(TAG) \
		--region=$(REGION) \
		--platform=managed \
		--allow-unauthenticated \
		--memory=2Gi \
		--set-env-vars=MODEL_SOURCE=gcs,GCS_MODEL_BUCKET=$(GCP_BUCKET),GCS_MODEL_PATH=models

## Get deployed API URL
get-api-url:
	@gcloud run services describe rnn-api --region=$(REGION) --format='value(status.url)'

## Download trained model from GCS to local
download-model:
	gsutil -m cp "gs://$(GCP_BUCKET)/models/model_*.pth" models/
	@echo "Downloaded models from GCS to local models/ directory."



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
