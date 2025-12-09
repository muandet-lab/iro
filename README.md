# iro — Imprecise Risk Optimization in Python

![PyPI version](https://img.shields.io/badge/PyPI-not%20published-yellow)
[![Documentation Status](https://readthedocs.org/projects/iro/badge/?version=latest)](https://iro.readthedocs.io/en/latest/)

**iro** is a Python library for Imprecise Risk Optimization (IRO) which provides tools for risk aggregation and optimization under uncertainty. **iro** leverages an imprecise learning setup to allow users to state their generalisation preferences at test-time.

#### Reference
Our IRO framework is, at its core, based on the ICML'24 paper "Domain Generalisation via Imprecise Learning".

> [*"Domain Generalisation via Imprecise Learning."* ICML2024.](https://arxiv.org/abs/2404.04669) (Anurag Singh, Siu Lun Chau, Shahine Bouabid, Krikamol Muandet)

## Features

Conditional Value at Risk (CVaR) is the base risk measure for iro. The framework is however designed to be risk measure-agnostic and a core feature of iro is the implementation of further risk measures, (as of now) including

* [Exponential Spectral Risk Measure](https://doi.org/10.48550/arXiv.1103.5409) which takes a risk aversion parameter value at test-time, preceded by an imprecise training flow.

See https://en.wikipedia.org/wiki/Coherent_risk_measure and https://doi.org/10.1111/1467-9965.00068 for more information regarding risk measures.

## Examples

The `examples` module contains extended experiments on the Colored MNIST benchmark from the paper's original repository. There are also examples which demonstrate the behavior of risk aggregation functions when applied to both synthetically generated data as well as [Fisher's Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) and [UCI Default of Credit Card Clients.](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).


## Installation

The package is **not yet published on PyPI**. We plan to reserve the PyPI name `iropy` for future releases.  

For now, you can install directly from this GitHub repository:

```bash
git clone https://github.com/muandet-lab/iro.git
cd iro
pip install -e .
```

Relevant source material for iro includes, but is not limited to:

* [Domain Generalisation via Imprecise Learning](https://doi.org/10.48550/arXiv.2404.04669)
* [Exponential Spectral Risk Measures](https://doi.org/10.48550/arXiv.1103.5409)
* [Fairness Risk Measures](https://doi.org/10.48550/arXiv.1901.08665)
* [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://doi.org/10.48550/arXiv.2012.07421)
