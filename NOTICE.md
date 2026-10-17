# Third-party code and data

* `experiments/tsr/tslib/` vendors backbone models and layers from
  [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT License,
  `third_party/licenses/Time-Series-Library-MIT.txt`); only import paths were changed.
* `experiments/tsr/tslib/models/PatchTSTo.py`, `layers/PatchTSTo_backbone.py`, `layers/PatchTSTo_layers.py` and
  `layers/RevINo.py` are the original PatchTST implementation ([PatchTST](https://github.com/yuqinie98/PatchTST),
  Apache License 2.0, `third_party/licenses/PatchTST-Apache-2.0.txt`), as distributed with OnlineTSF; import paths and
  the forward signature were changed.
* The published baselines (online gradient descent and Proceed) are not redistributed. `scripts/get_baselines.sh`
  downloads the public [DynaME repository](https://github.com/shhong97/DynaME) (which contains the
  [OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF) implementations of GD and Proceed) at commit `0743ad6` and
  applies `third_party/dyname_dump_preds.patch`, which adds a dump of the forecasts, replaces the removed `np.Inf` alias
  and adds a stub for `wandb`.
* The tuning-free delayed learners of Flaspohler et al. (2021) are re-implemented in `experiments/replay2.py`
  (`replay_poold`); the re-implementation reproduces the weights of the authors' PoolD library to 1e-15.
* No data are redistributed. CME futures data (Databento) are licensed; the scripts in `experiments/finance_data/`
  need your own API key.
