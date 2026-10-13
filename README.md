# neural-networks-assignment
## A1
### Baseline Model Architecture
```mermaid
flowchart TD
    A[Month] --> E1[Embedding<br/>dim = 8]
    B[Town] --> E2[Embedding<br/>dim = 8]
    C[Flat Model Type] --> E3[Embedding<br/>dim = 8]
    D[Storey Range] --> E4[Embedding<br/>dim = 8]

    E1 --> CAT[Concatenate]
    E2 --> CAT
    E3 --> CAT
    E4 --> CAT

    CONT[6 Continuous Features] --> CAT

    CAT -->|38 features| L1[Linear<br/>38 → 64]
    L1 --> LN[LayerNorm<br/>64]
    LN --> R[ReLU]
    R --> L2[Linear<br/>64 → 1]
    L2 --> S[Standardized Price Prediction]
    S --> DS[Rescale<br/>× target_std + target_mean]
    DS --> OUT[Predicted Resale Price]
```

### Baseline Parameter Count
The baseline model has **3,505 trainable parameters**.
#### Embedding Layers
The four categorical features have cardinalities of 13, 27, 44, and 18, with an embedding dimension of 8 for each feature.
`13(8) + 27(8) + 44(8) + 18(8)`
`= 104 + 216 + 352 + 144`
`= 816`
#### First Linear Layer
The four 8-dimensional embeddings produce 32 features. Together with the 6 continuous features, the input dimension is 38.
The first linear layer maps 38 inputs to 64 hidden units:
`38(64) + 64 = 2496`
where 38x64 represent 38 weight parameters for each sample multiplied by input width 64, and 64 additional bias parameters.
#### Layer Normalization
Layer normalization contains one learnable scale and one learnable bias for each of the 64 hidden units:
`64 + 64 = 128`
#### Output Layer
The output layer maps the 64 hidden units to one output with 64 weight parameters:
`64(1) + 1 = 65`
### Total Trainable Parameters
`816 + 2496 + 128 + 65 = 3505`
### Hyperparameter Optimisation
BEST CONFIGURATION:\
Hidden width: 128\
Embedding dimension: 12\
Best validation RMSE: 59,766.91

    Hidden    Embedding      Best Val RMSE
       128           12          59,766.91
       256            2          59,926.48
       256            4          61,435.03
       128            4          61,977.41
       256           16          61,999.66
        32           12          62,479.79
        16           12          62,654.96
        32            4          62,886.20
        64           16          63,032.88
        64            8          63,422.46
        64            2          63,491.59
       256            8          63,806.88
        64           12          64,105.07
        32           16          64,593.14
        64            4          64,633.77
       256           12          65,072.00
       128           16          65,599.97
        32            8          66,199.89
       128            2          66,327.64
        16            2          66,425.56
       128            8          66,832.43
        16           16          67,164.91
        16            8          67,520.02
        16            4          68,879.52
        32            2          69,203.99
### Training and Validation RMSE
![Training and Validation RMSE](https://github.com/malepsyche/neural-networks-assignment/blob/main/hdb-price-prediction-assignment-main/outputs/tuning/best_learning_curve.png)
### Retrain model on combined train and validation sets:
![Training and Validation RMSE](https://github.com/malepsyche/neural-networks-assignment/blob/main/hdb-price-prediction-assignment-main/outputs/final/final_learning_curve.png)
    epoch 01 | train RMSE 44,383.07 | test RMSE 107,977.41
    epoch 02 | train RMSE 43,615.70 | test RMSE 93,158.23
    epoch 03 | train RMSE 40,706.75 | test RMSE 104,607.33
    epoch 04 | train RMSE 40,913.82 | test RMSE 97,495.47
    epoch 05 | train RMSE 39,331.23 | test RMSE 99,816.12
    epoch 06 | train RMSE 39,456.30 | test RMSE 104,240.83
    epoch 07 | train RMSE 39,469.28 | test RMSE 105,467.51
    epoch 08 | train RMSE 39,193.23 | test RMSE 99,124.45
    epoch 09 | train RMSE 38,682.61 | test RMSE 100,854.71
    epoch 10 | train RMSE 39,392.42 | test RMSE 92,690.02
    epoch 11 | train RMSE 39,049.87 | test RMSE 92,375.76
    epoch 12 | train RMSE 38,113.98 | test RMSE 102,521.88
    epoch 13 | train RMSE 38,359.75 | test RMSE 91,237.82
FINAL RESULTS
Hidden width: 128
Embedding dimension: 12
Epochs: 13
Test RMSE: 91,237.82
Test R²: 0.7128