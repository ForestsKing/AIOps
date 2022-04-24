This project realizes the anomaly detection and root cause location of multimodal data. The anomaly detection part
adopts **MTAD-GAT** model (metric, trace) and **DeepLog** model (log), The root cause localization part adopts the **Squeeze** model.

# Experimental setup

- Train:
    - Train data (update model): **2022-03-24 15:20:00** ~ **2022-03-25 08:06:00**
    - Valid data (prevent overfitting): **2022-03-25 08:07:00** ~ **2022-03-25 15:19:00**
- Test:
    - Valid data (search threshold): **2022-03-26 08:30:00** ~ **2022-03-26 11:29:00**
    - Test data (evaluation model): **2022-03-26 11:30:00** ~ **2022-03-26 20:29:00**

# Result
## anomaly detection

|                 |     | P    | R    | F1     |
|-----------------|-----|------|------|--------|
| metric          | -   | 0.5329 | 0.7945 | 0.6379 |
| metric          | +   | 0.8873 | 0.7412 | 0.8077 |
| trace           | -   | 0.1943 | 0.3527 | 0.2506 |
| trace           | +   | 0.2073 | 0.8706 | 0.3348 |
| log             | -   | 0.1382 | 0.4027 | 0.2058 |
| log             | +   | 0.1759 | 1.0000 | 0.2992 |
| metric+trace    | -   | 0.3190 | 0.6218 | 0.4217 |
| metric+trace    | +   | 0.7917 | 0.8941 | 0.8398 |
| metric+trace+log | -   |0.3347|0.6359| 0.4386 |
| metric+trace+log | +   |0.8085|0.8941| 0.8492 |

## root cause localization

| |     |PR@1|PR@2|PR@3|PR@4|PR@5|PR@Avg|
|----|-----|----|----|----|----|----|----|
|RootCause| -   |0.2783|0.4001|0.5192|0.5953|0.6217|0.4829|
|RootCause| +   |0.5739|0.7652|0.8522|0.9217|0.9391|0.8104|

# More

- See [Log](./log.md) for training and testing logs.
- See [Loss](./result/img/) for loss visualization.
