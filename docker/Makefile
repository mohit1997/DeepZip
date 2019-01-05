help:
	@cat Makefile

DATA?="${HOME}/Data"
#GPU?=0
DOCKER_FILE=Dockerfile
#DOCKER=GPU=$(GPU) nvidia-docker
DOCKER=docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
#CUDA_VERSION?=9.0
#CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t keras --build-arg python_version=$(PYTHON_VERSION)  -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

