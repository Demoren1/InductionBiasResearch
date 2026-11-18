# What the hard indexer omits

Oracle diagnostic on the already evaluated source-count test panels. The teacher is used only to audit ideal representations. These are not learned-method results. No training or selection is performed.

| Rank | True continuous coefficients | Exact equality groups | Rank of one-support equality system | Unresolved equality-code directions | Ideal equality MSE | Ideal continuous-map MSE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 3 | 0 | 2.2289911e-15 | 2.2289911e-15 |
| 2 | 6 | 204 | 68 | 136 | 1.2752087 | 1.3964229e-15 |
| 5 | 15 | 204 | 68 | 136 | 0.63829141 | 8.5947034e-16 |

One support observation has 68 scalar outputs. The exact equality indexer has 204 independent codes at ranks 2/5; 136 directions remain unidentified. The true continuous maps need only 6/15 coefficients, and their support matrices have full column rank on every audited task.

Projection error measures false equality constraints in compressed selected indices. Its population-MSE normalization assumes the generator's independent standard-normal coefficients and inputs. Zero projection error means the representation contains all true functions, not that one support observation can identify their codes.

The additional support-count diagnostic reserves the same last 15 query observations for support counts 1/3/5. It is a distinct diagnostic protocol, not a replacement for the primary one-support comparison. Its numerical data are in representation_audit.json.

## Interpretation

For example, weights `a`, `b`, and `(a+b)/2` are generically distinct. Their exact equality indexer has three separate codes and does not specify the relation between them. A linear map can express the same family using only two task coefficients.

The ideal equality model above contains every true task. Its reported error is the error of minimum-norm least squares from one support observation, not a representation error or a lower bound for every possible adaptation algorithm. A method that additionally learns correlations between its codes could improve it, but those correlations are additional structure beyond the indexer.

The actual selected indices confirm two different effects:

| Selected index, n=200 | Groups | Generator energy excluded by equality constraints | One-support test MSE |
|---|---:|---:|---:|
| Rank 2, native G34 | 15 | 29.56% | 0.783069 |
| Rank 2, native Pair | 70 | 42.97% | 1.431133 |
| Rank 2, fresh G34 | 70 | 7.89% | 0.656581 |
| Rank 2, fresh Pair | 204 | 0% | 1.275204 |
| Rank 5, native G34 / Pair | 204 | 0% | 0.755192 / 0.755195 |
| Rank 5, fresh G34 / Pair | 204 | 0% | 0.638291 |

Compressing distinct weights into one signed code imposes false equalities; keeping all 204 codes avoids that approximation error but leaves the continuous dependencies unspecified. Different selected partitions need not have monotonically decreasing projection error as their group count grows.

With the same ideal 204-code local indexer and five support observations, minimum-norm least squares reaches MSE 3.70e-15 at rank 2 and 1.85e-15 at rank 5 on the common 15-query diagnostic. With three support observations the errors are 2.98e-11 and 2.05e-10. Thus the exact equality model can fit these tasks; the limitation demonstrated here is what can be inferred from one support observation without additional relationships between codes.
