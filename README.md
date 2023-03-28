[![Actions Status](https://github.com/pln-fing-udelar/fast-krippendorff/workflows/CI/badge.svg)](https://github.com/pln-fing-udelar/fast-krippendorff/actions)
[![Version](https://img.shields.io/pypi/v/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)
[![License](https://img.shields.io/pypi/l/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/krippendorff.svg)](https://pypi.python.org/pypi/krippendorff)

# Fast Krippendorff

Fast computation of [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) agreement measure.

Based on [Thomas Grill implementation](https://github.com/grrrr/krippendorff-alpha).

## Example usage

Given a reliability data matrix, run:

```python
import krippendorff

krippendorff.alpha(reliability_data=...)
```

See `sample.py` and `alpha`'s docstring for more details.

## Installation

```bash
pip install krippendorff
```

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
