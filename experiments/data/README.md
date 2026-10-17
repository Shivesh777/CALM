# Data

The harness reads the datasets from this folder (`--data-root data`), under the file names used by
`tsr/data.py`. The benchmarks are the standard public versions used by the long-term forecasting literature:

| file | dataset | source |
| --- | --- | --- |
| `ETTh2.csv`, `ETTm1.csv` | Electricity Transformer Temperature | https://github.com/zhouhaoyi/ETDataset |
| `weather.csv` | Weather | the Autoformer / Time-Series-Library dataset collection |
| `electricity.csv` | Electricity (ECL) | the Autoformer / Time-Series-Library dataset collection |
| `traffic.csv` | Traffic | the Autoformer / Time-Series-Library dataset collection |

The finance panels are built by the scripts in `../finance_data/`: the crypto panels from public Binance bars, the CME
futures panels from Databento bars, which are licensed and are not distributed here.
