SRC_DIR=./turbulence_statistics

lint:
	black ${SRC_DIR} # ${TEST_DIR}
	mypy ${SRC_DIR} # ${TEST_DIR}

install:
	pip install .

install_dev:
	pip install -e .[dev]
