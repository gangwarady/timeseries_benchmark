PyTorch fundamentals

Benchmarked three architectures on KPI time-series anomaly detection.

| Model | MSE | MAE | Params |
|-------|-----|-----|--------|
| FeedForward | 4.37 | 1.65 | 2,817 |
| LSTM | 2.36| 1.20 | 54,849 |
| Transformer | 0.26 | 0.31 | 76,161 |

Key finding: FeedForward can't detect trends (no time dimension).
LSTM and Transformer both learn the KPI_0 trend signal.
Transformer converges faster due to parallel attention.
