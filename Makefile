# =============================================================================
# Makefile — thin wrappers around scripts/dev.sh
#
# Prerequisites:
#   podman on PATH  OR  running inside a toolbox (flatpak-spawn bridge used)
#
# Usage:
#   make build        # build the dev image
#   make shell        # interactive dev shell
#   make test         # full test suite (JIT disabled)
#   make test-fast    # test suite with pytest-xdist (-n auto)
#   make lint         # ruff check + format check
#   make typecheck    # mypy
#   make runtime      # build the production runtime image
#   make save         # export runtime image → rotmd.tar (for Apptainer)
# =============================================================================

SHELL := /bin/bash
DEV   := ./scripts/dev.sh

.PHONY: build shell test test-fast lint typecheck runtime save help

build:
	$(DEV) build

shell:
	$(DEV) shell

test:
	$(DEV) test

test-fast:
	$(DEV) test-fast

lint:
	$(DEV) lint

typecheck:
	$(DEV) typecheck

runtime:
	$(DEV) build-runtime

save:
	$(DEV) save-runtime

help:
	@$(DEV) help
