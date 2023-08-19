.PHONY: clean lint requirements format sort gen-global-requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = next-watch-teste
PYTHON_INTERPRETER = python3.9
CONDA_FOLDER_NAME = conda

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
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.minimal
	$(PYTHON_INTERPRETER) -m pip install -r requirements.local
	sudo curl -o ./assets/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
	sudo curl -o ./assets/aws-java-sdk-bundle-1.12.506.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.506/aws-java-sdk-bundle-1.12.506.jar
	sudo curl -o ./assets/hadoop-common-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.4/hadoop-common-3.3.4.jar
	sudo curl -o ./assets/postgresql-42.6.0.jar https://jdbc.postgresql.org/download/postgresql-42.6.0.jar
	#cp ./assets/hadoop-aws-3.3.4.jar ~/$(CONDA_FOLDER_NAME)/envs/next-watch/lib/python3.10/site-packages/pyspark/jars/
	#cp ./assets/aws-java-sdk-bundle-1.12.506.jar ~/$(CONDA_FOLDER_NAME)/envs/next-watch/lib/python3.10/site-packages/pyspark/jars/
	#cp ./assets/hadoop-common-3.3.4.jar ~/$(CONDA_FOLDER_NAME)/envs/next-watch/lib/python3.10/site-packages/pyspark/jars/

## Generate requirements for distributable packages.jar
environment:
	pip list --format=freeze > requirements.dist

## Config databases, init programs, etc...
init:
	docker compose build --no-cache
	docker compose up postgres create-databases
	cd src
	python3.9 scripts/create_aws_credentials_file.py
	cd ..
	#docker compose up airflow-init
	#docker compose down --volumes --remove-orphans 

## Run DE pipelines
run-de:
	docker compose exec dev-spark bash -c "cd src; python3.9 main.py -p 'de'"

## Populate Databse with Users from production datasets
pop-db-users:
	docker compose exec dev-spark bash -c "cd src; python3.9 scripts/populate_db_with_users.py"

## Run DS pipelines
run-ds:
	docker compose exec dev-spark bash -c "cd src; python3.9 main.py -p 'ds'"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Format with Black
format:
	black src

## Sort imports using iSort
sort:
	isort src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.9
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

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
