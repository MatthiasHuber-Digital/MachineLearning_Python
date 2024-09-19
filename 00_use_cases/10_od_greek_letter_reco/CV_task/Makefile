# Set default variables
PACKAGE_NAME ?= simple_detector

# Declare phony targets
.PHONY: lint format

# Linting
lint:
	@mypy $(PACKAGE_NAME)
	@ruff $(PACKAGE_NAME)

# Formatting
format:
	@isort $(PACKAGE_NAME)
	@black $(PACKAGE_NAME)