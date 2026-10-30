# Binary interchange, version 1

Integers are unsigned 32-bit little-endian. Values are IEEE-754 little-endian
float32 with no padding. Files are limited to 256 MiB. The loader rejects trailing
bytes, truncation, and non-finite values, and checks dimensions before allocation.

## Model

| Byte offset | Field |
| --- | --- |
| 0 | Eight ASCII bytes `DMLP0001` |
| 8 | Input dimension `I` |
| 12 | Hidden dimension `H` |
| 16 | Output dimension `O` |
| 20 | First weights, `H*I` floats in PyTorch `[output][input]` order |
| next | First biases, `H` floats |
| next | Second weights, `O*H` floats in `[output][input]` order |
| next | Second biases, `O` floats |

Length is `20 + 4*(I*H + H + H*O + O)`. Dimensions are in [1,1,000,000], and the
payload must fit the file-size limit. ReLU follows the first layer. C++ transposes
the weights once on loading into `[input][output]` order, shared by both kernels.
Integer products are checked for overflow.

## Matrix (inputs or scores)

| Byte offset | Field |
| --- | --- |
| 0 | Eight ASCII bytes `DMLPMAT1` |
| 8 | Rows `N` (zero allowed) |
| 12 | Columns `C` (positive, at most 1,000,000) |
| 16 | `N*C` row-major float32 values |

Length is `16 + 4*N*C`. Input columns must match the model input dimension.
Output columns match its output dimension. Empty matrices still need a valid
column count. MNIST rows contain 784 pixels scaled by 1/255; output rows contain
ten logits in digit order 0–9.

`model.pt` is a PyTorch state dictionary for the Python reference, loaded using
`weights_only=True`. C++ uses only `model.bin` and matrix files.
