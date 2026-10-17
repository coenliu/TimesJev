#!/bin/bash
# Download the TSQA files used here (anomaly detection, classification, open-ended QA for MCQA and true/false QA) and
# the two Decider backbones, at the revisions used in our experiments.
# TSQA is gated on Hugging Face: accept its terms on https://huggingface.co/datasets/Time-MQA/TSQA and run
# `hf auth login` first.
set -e
cd "$(dirname "$0")/.."
hf download Time-MQA/TSQA --repo-type dataset --revision a7f9e95667b695d2647b9eff92bb6a38b5183a34 \
  --local-dir data/TSQA --include "Anomaly_Detection/*" "Classification/*" "Open_Ended_QA/*" "README.md"
hf download Mapika/decider-2b --revision 533964dae8be954c5b5e19fa4948e48408094c1e
hf download Mapika/decider-4b --revision eb5fbdfc9448473ec25e399882912863afbdb70e
