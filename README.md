# Salamander

[![Python versions supported][python-image]][python-url]
[![License][license-image]][license-url]
[![Code style][style-image]][style-url]

[python-image]: https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg
[python-url]: https://github.com/parklab/salamander
[license-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/parklab/salamander/blob/main/LICENSE
[style-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/psf/black

Salamander is a non-negative matrix factorization (NMF) framework for signature analysis build on top of [AnnData](https://anndata.readthedocs.io/en/latest/) and [MuData](https://mudata.readthedocs.io/en/latest/). It implements multiple NMF algorithms, common visualizations, and can be easily customized & expanded.

---

## Installation

PyPI:
```bash
pip install salamander-learn
```

## Usage

The following example illustrates the basic syntax:

```python
import anndata as ad
import salamander as sal

# initialize data
adata = ad.AnnData(...)

# NMF with Poisson noise
model = sal.models.KLNMF(n_signatures=5)
model.fit(adata)

# barplot
model.plot_signatures()

# stacked barplot
model.plot_exposures()

# signature correlation
model.plot_correlation()

# sample_correlation
model.plot_correlation(data="samples")

# dimensionality reduction of the exposures
model.plot_embeddings(method="umap")
```

For examples of how to customize any NMF algorithm and the plots, check out [the tutorial](https://github.com/parklab/salamander/blob/main/tutorial.ipynb). The following algorithms are currently available:
* [NMF with KL-divergence loss](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)
* [minimum-volume NMF](https://browse.arxiv.org/pdf/1907.02404.pdf)
* [a variant of correlated NMF](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=87224164eef14589b137547a3fa81f06eef9bbf4)

## License

MIT

## Changelog

Consult the [CHANGELOG](https://github.com/parklab/Salamander/blob/main/CHANGELOG.md) file for enhancements and fixes of each version.
