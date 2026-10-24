# All logic lives in tasks.py so the Windows shim and `make` cannot drift apart.
PY ?= python

.PHONY: test pilot gate1 results clean help

help:
	@$(PY) tasks.py help

test:
	@$(PY) tasks.py test

pilot:
	@$(PY) tasks.py pilot

gate1:
	@$(PY) tasks.py gate1

results:
	@$(PY) tasks.py results

clean:
	@$(PY) tasks.py clean
