# iro — Imprecise Risk Optimization in Python

![PyPI version](https://img.shields.io/badge/PyPI-not%20published-yellow)
[![Documentation Status](https://readthedocs.org/projects/iro/badge/?version=latest)](https://iro.readthedocs.io/en/latest/)

**iro** is a Python library for Imprecise Risk Optimization (IRO), providing tools for risk aggregation and optimization under uncertainty. It implements ideas from the domain generalization literature, enabling robust learning across multiple environments.

* Free software: MIT License

#### Reference
Our IRO framework is, at its core, based on the ICML'24 paper "Domain Generalisation via Imprecise Learning".

> [*"Domain Generalisation via Imprecise Learning."* ICML2024.](https://arxiv.org/abs/2404.04669) (Anurag Singh, Siu Lun Chau, Shahine Bouabid, Krikamol Muandet)

## Features

Currently implemented aggregation functions for risk include:

- **Conditional Value-at-Risk (CVaR)**  
- **Value-at-Risk (VaR)**  
- **Entropic risk measure**  
- **Mean (expected risk)**  
- **Worst-case risk**  
- **Median (robust alternative to mean)**  
- **Variance (penalizing spread)**
- **Proportional hazard risk measure**
- **Wang risk measure**

* See https://en.wikipedia.org/wiki/Coherent_risk_measure and https://doi.org/10.1111/1467-9965.00068 for more information regarding risk measures.

## Examples
The `examples` module contains examples which demonstrate the behavior of risk aggregation functions when applied to both synthetically generated data as well as [Fisher's Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) and [UCI Default of Credit Card Clients.](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)


## Installation

The package is **not yet published on PyPI**. We plan to reserve the PyPI name `iropy` for future releases.  

For now, you can install directly from this GitHub repository:

```bash
git clone https://github.com/muandet-lab/iro.git
cd iro
pip install -e .
