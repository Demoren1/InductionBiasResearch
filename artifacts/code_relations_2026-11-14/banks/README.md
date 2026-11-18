# Using the recovered bank with linear code relations

Each `rank*_n*/g34.npz` or `pair.npz` contains:

- `index`: a signed integer matrix of shape `(68, 70)`, with axes `(output, input)`. Zero marks an absent weight; absolute value selects a code, starting at one.
- `code_map`: a matrix mapping task-specific latent coefficients to bank values. Its shapes are `(3, 3)`, `(204, 6)`, and `(204, 15)` for ranks 1, 2, and 5.
- `ridge`: the regularization chosen on validation. All these selected banks use zero.

The index and map are shared between tasks. A new task fits only 3, 6, or 15 coefficients using its support input and output. The following example uses a single support observation, whose output has 68 components:

```python
import numpy as np

bank = np.load("pair.npz")
index = bank["index"]
code_map = bank["code_map"]
dimension = code_map.shape[1]

# Expand the map into a basis for the linear layer's weights.
dictionary = np.vstack([np.zeros((1, dimension)), code_map])
basis = np.sign(index)[..., None] * dictionary[np.abs(index)]

# x_support: (70,), y_support: (68,). No query labels enter this fit.
design = np.einsum("i,oik->ok", x_support, basis)
latent = np.linalg.lstsq(design, y_support, rcond=1e-12)[0]

# x_new: (70,); prediction: (68,).
prediction = np.einsum("i,oik,k->o", x_new, basis, latent)
```

These maps were learned from the existing source training functions: 400 functions at rank 1 and 800 at ranks 2/5, with 20, 80, or 200 observations per function. No teacher weights or generator mixing enter fitting. The method refits source outputs on the index's active inputs, averages signed weights within codes, and takes an uncentered SVD in the weight-space metric. The full source/validation procedure is in `experiments/msr_reproduction_v1/code_relations.py`.

The banks are specific to the synthetic environments in this experiment. Learning their code relations is a transfer step across multiple source functions, rather than recovery of the whole linear map from one dense layer.
