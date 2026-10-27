# CPU latency benchmark (batch 1, 4 threads, 50 runs)

| Run | Runtime | Params (M) | Size (MB) | Median (ms) | p95 (ms) |
|---|---|---|---|---|---|
| custom_cnn | pytorch | 0.39 | 1.58 | 11.21 | 15.48 |
| mobilenet_v2 | pytorch | 2.24 | 9.16 | 16.44 | 18.89 |
| mobilenet_v2 | onnxruntime | 2.24 | 8.91 | 2.5 | 3.88 |
| mobilenet_v2_headonly | pytorch | 2.24 | 9.16 | 15.12 | 16.98 |
| mobilenet_v2_noaug | pytorch | 2.24 | 9.16 | 14.86 | 16.68 |
| resnet18 | pytorch | 11.18 | 44.79 | 22.42 | 28.12 |
