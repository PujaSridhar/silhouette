VENV ?= .venv
PY := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: help build parity test outfit critique gaps week rules demo clean sensitivity measure

build:  ## regenerate the published pages from templates + engine exports
	$(PY) tools/export_demo_data.py
	PYTHONPATH=tools $(PY) tools/export_app_data.py
	$(PY) tools/build_pages.py

parity: ## replay both built pages' parity fixtures in Node, and the Rate tab flow (run build first)
	node tools/check_parity.js demo/index.html app/index.html
	node tools/check_rate_flow.js app/index.html

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t24

test:   ## run the full suite
	$(PY) -m pytest tests/ -q

outfit: ## best outfits for a context (CONTEXT=office)
	$(PY) -m silhouette.cli outfit --context $(or $(CONTEXT),office) --top 3

gaps:   ## which single purchase buys the most
	$(PY) -m silhouette.cli gaps

week:   ## plan a week under wear limits
	$(PY) -m silhouette.cli week

rules:  ## list the rule set
	$(PY) -m silhouette.cli rules -v

demo:   ## the phase 0 walkthrough
	@echo "\n=== 1. the wearer's own signature outfit ==="
	@$(PY) -m silhouette.cli critique --context weekend \
	  --items top_crop_rib_ivory,bot_wide_blk,shoe_sneaker_white
	@echo "\n=== 2. the same two garments, untucked then tucked ==="
	@$(PY) -m silhouette.cli critique --context office \
	  --items top_blouse_struct_cream,bot_wide_cream,shoe_heel_blk_block
	@$(PY) -m silhouette.cli critique --context office --tucked \
	  --items top_blouse_struct_cream,bot_wide_cream,shoe_heel_blk_block
	@echo "\n=== 3. throwing the rise away ==="
	@$(PY) -m silhouette.cli critique --context campus \
	  --items top_hoodie_oversized_grey,bot_wide_cream,shoe_sneaker_white
	@echo "\n=== 4. what to buy ==="
	@$(PY) -m silhouette.cli gaps --top 4
	@echo "\n=== 5. the week, and the bottleneck ==="
	@$(PY) -m silhouette.cli week

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

bootstrap: ## synthetic labels to validate the harness (not evidence)
	$(PY) -m silhouette.cli bootstrap --n 180 --seed 7

label:  ## rate outfits against the rubric
	$(PY) -m silhouette.cli label --n 20

evals:  ## scoreboard against labels (SOURCE=expert|bootstrap)
	$(PY) -m silhouette.cli eval --source $(or $(SOURCE),expert)

fit:    ## fit rule weights to labels
	$(PY) -m silhouette.cli fit --source $(or $(SOURCE),expert)

coverage: ## labels per rule (SOURCE=expert|bootstrap|all)
	$(PY) -m silhouette.cli coverage --source $(or $(SOURCE),expert)

sensitivity: ## how precise volume needs to be (W1 target)
	$(PY) -m silhouette.vision.cli sensitivity --seeds $(or $(SEEDS),200)

measure: ## measured vs hand-judged volume on photos/ (MASKS=masks|masks_model)
	$(PY) -m silhouette.vision.cli measure --masks $(or $(MASKS),masks)

demo-serve: ## serve the interactive demo
	$(PY) -m http.server 8787 --directory demo
