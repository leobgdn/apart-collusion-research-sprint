# All logic lives in tasks.py so the Windows shim and `make` cannot drift apart.
PY ?= python

.PHONY: test gate1 dry_pilot check_server throughput pilot gate2_data results clean help

help:
	@$(PY) tasks.py help

test gate1 dry_pilot check_server throughput pilot gate2_data results clean:
	@$(PY) tasks.py $@
