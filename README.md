# Krippendorff

Fast computation of [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) agreement measure.

Based on [Thomas Grill implementation](https://github.com/grrrr/krippendorff-alpha). Works on Python 3+.

## Example usage

Given a `value_count` matrix, which has the value count for each unit:

```python
import krippendorff

krippendorff.alpha(value_count)
```

See `sample.py` and `alpha` docstring for more.

## Installation

```shell
pip install krippendorff
```

## Caveats

The implementation is fast as it doesn't do a nested loop for the coders. However, V should be small, since a matrix of VxV it's used.
