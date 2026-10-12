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
**Total trainable parameters: 3,505**