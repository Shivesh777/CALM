#!/usr/bin/env bash
# Fetch the public code of the published baselines (online gradient descent and Proceed, as contained in the DynaME
# repository, which builds on OnlineTSF) at the commit used in the paper, and add the forecast dump that lets their
# forecasts enter the pools as experts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d third_party/DynaME ]; then
  git clone https://github.com/shhong97/DynaME.git third_party/DynaME
fi
cd third_party/DynaME
git checkout -q 0743ad664163262f78579572767ce66a6aa74ab5
if git apply --check ../dyname_dump_preds.patch 2>/dev/null; then
  git apply ../dyname_dump_preds.patch
fi
echo "Baseline code ready in third_party/DynaME. Run it with PYTHONPATH=third_party/DynaME/stubs and DUMP_PREDS=<prefix>;"
echo "see experiments/pre/*.sh for the commands and released hyper-parameters of each dataset."
