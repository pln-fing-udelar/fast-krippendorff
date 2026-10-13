[![Actions Status](https://github.com/pln-fing-udelar/fast-krippendorff/workflows/CI/badge.svg)](https://github.com/pln-fing-udelar/fast-krippendorff/actions)
[![Coverage](https://codecov.io/gh/pln-fing-udelar/fast-krippendorff/branch/main/graph/badge.svg)](https://codecov.io/gh/pln-fing-udelar/fast-krippendorff)
[![Version](https://img.shields.io/pypi/v/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)
[![License](https://img.shields.io/pypi/l/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)

# Fast Krippendorff

Fast computation of [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) agreement measure.

Based on [Thomas Grill implementation](https://github.com/grrrr/krippendorff-alpha).

## Installation

```bash
pip install krippendorff
```

## Example usage

Given a reliability data matrix (coders as rows, units as columns), run:

```python
import krippendorff

krippendorff.alpha(reliability_data=...)
```

### Using with Pandas DataFrames

Pandas is commonly used to load annotations from CSV or database exports. Because `krippendorff.alpha` expects the reliability data matrix to have shape `(M, N)` (where `M` is the number of coders/raters and `N` is the number of units/items), handle your DataFrame as follows:

#### 1. Wide format (Units as rows, Coders as columns)

When each row represents an annotated item and each column is an annotator (with missing annotations represented by `NaN` or `None`):

```python
import pandas as pd
import krippendorff

# DataFrame where columns are coders and rows are units
df = pd.DataFrame({
    "coder_1": [3, 1, 2, None],
    "coder_2": [3, 2, 2, 4],
    "coder_3": [4, 1, None, 4],
})

# Transpose so rows are coders and columns are units:
alpha = krippendorff.alpha(reliability_data=df.to_numpy().T, level_of_measurement="interval")
print("Krippendorff's alpha:", alpha)
```

#### 2. Long format (Annotations table: `unit_id`, `coder_id`, `score`)

When annotations are stored as one row per rating:

```python
import pandas as pd
import krippendorff

df = pd.DataFrame({
    "unit_id": [1, 1, 2, 2, 2, 3, 3],
    "coder_id": ["A", "B", "A", "B", "C", "A", "C"],
    "score": [4, 4, 2, 3, 2, 5, 5],
})

# Pivot into a matrix with coders as rows and units as columns:
matrix = df.pivot(index="coder_id", columns="unit_id", values="score").to_numpy()
alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
print("Krippendorff's alpha:", alpha)
```

### Bootstrap Confidence Intervals and Significance Testing

Compute bootstrap confidence intervals, standard errors, and Krippendorff's $q$-value ($P(\alpha^* < \alpha_{\text{min}})$) with high-speed vectorized resampling:

```python
import krippendorff

res = krippendorff.bootstrap_ci(
    reliability_data=matrix,
    level_of_measurement="interval",
    confidence_level=0.95,
    n_resamples=2000,
    alternative_threshold=0.800,  # Krippendorff q-value against minimum reliability benchmark
    random_state=42,
)

print(f"Alpha: {res.alpha:.3f}")
print(f"95% CI: [{res.ci_low:.3f}, {res.ci_high:.3f}]")
print(f"Standard Error: {res.standard_error:.4f}")
print(f"q-value (P(alpha < 0.8)): {res.q_value:.4f}")
```

See `example.py` and function docstrings for additional examples and options.

## Caveats

The implementation is fast as it doesn't do a nested loop for the coders. However, `V` should be small, since a `VxV` matrix it's used.

## Citing

If you use this code in your research, please cite Fast Krippendorff:

```bibtex
@misc{castro-2017-fast-krippendorff,
  author = {Santiago Castro},
  title = {Fast {K}rippendorff: Fast computation of {K}rippendorff's alpha agreement measure},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pln-fing-udelar/fast-krippendorff}}
}
```
