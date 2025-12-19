## Models we may want to use
### Isolation forest
This is the most basic model, similar to a decision tree and a good starting point. 

### Auto-Encoders
Auto encoders trained on normal network traffic will fail to reconstruct abnormal network data effectively. This reconstruction error (difference between the input vector and the reconstructed output vector) is the anomaly signal, with higher error correlating to higher likelihood of attack traffic.