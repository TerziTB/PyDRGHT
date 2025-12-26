# PyDRGHT
![PyPI](https://img.shields.io/pypi/v/pydrght?color=blue)
![Python](https://img.shields.io/pypi/pyversions/pydrght)
![License](https://img.shields.io/pypi/l/pydrght)
![Code size](https://img.shields.io/github/languages/code-size/TerziTB/PyDRGHT)

*A comprehensive Python package for drought analysis, with standardized indices (SPI, SPEI, SSFI) and copula-based bivariate MSDI for multivariate drought characterization.*

PyDRGHT provides tools for **univariate and multivariate drought assessment**, combining standardized indices, classical indices, copula-based methods, and frequency analysis.

PyDRGHT is built for **monthly** hydrometeorological time series and supports any dataset that can be converted to a `pd.Series.`

---

## Features

- **Distribution Fitting Methods** 
  - PyDRGHT provides two independent frameworks for fitting probability distributions required in standardized drought indices (SPI, SPEI, SSFI, etc.):
  - Maximum Likelihood Estimation (MLE): Implemented using scipy.stats continuous distributions
  - Unbiased Probability Weighted Moments (UB-PWM): Based on L-moments and unbiased PWMs

- **Standardized indices** (based on SPI methodology with different variables):  
  - SPI (Standardized Precipitation Index)  
  - SPEI (Standardized Precipitation Evapotranspiration Index)  
  - SSFI (Standardized Streamflow Index)  
  - SGI (Standardized Groundwater Index)  
  - SSMI (Standardized Soil Moisture Index)

- **Classical drought indices**:  
  - RDI (Reconnaissance Drought Index)  
  - RAI (Rainfall Anomaly Index)  
  - PNI (Percent of Normal Index)  
  - CZI (China-Z Index)  
  - DI (Deciles Index)  

- **Multivariate drought indices**:  
  - MSDI (Multivariate Standardized Drought Index) – both **empirical** and **copula-based** approaches
  - TSDI (Trivariate Standardized Drought Index) - **empirical** approach

- **Drought characteristics**:  
  - Duration, severity, intensity, frequency  
  - Start and end of drought events  
  - Interarrival times between droughts  

- **Frequency analysis**:  
  - Univariate frequency analysis  
  - Copula-based BFA (Bivariate Frequency Analysis) 

- **Additional tools**:  
    - Potential Evapotranspiration (PET) via Hargreaves and Thornthwaite methods  
    - Dependence modeling with copulas (Archimedean, Elliptical, Extreme-value families) 
    - Advanced plotting utilities with automatic drought-severity shading, visually highlighting values below 0 (mild dryness), −1 (moderate drought), and −2 (severe drought). 

---

## Installation

```bash
pip install pydrght
```

Or from source:

```bash
git clone https://github.com/terzitb/pydrght.git
cd pydrght
pip install -e .
```

---
## Quickstart

```python
import pandas as pd
from pydrght import SI
from scipy.stats import gamma

# Load example precipitation data
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# 12-month SPI (parametric, 3-p Gamma distribution)
spi = SI(data["PRECIPITATION"], ts=12)
spi_param = spi.fit_parametric(gamma, is_2p=False)

print(spi_param.head())
```

---

## Package Structure

| Module / Subpackage        | Description                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `pydrght.BFA`            | **Bivariate Frequency Analysis** for drought severity & duration modeling                                        |
| `pydrght.CZI`            | **China-Z Index** (CZI)                                                                                         |
| `pydrght.DChar`          | **Drought Characteristics** (duration, severity, frequency, etc.)                                                     |
| `pydrght.DI`             | **Deciles Index** (DI)                                                                                          |
| `pydrght.Dist`           | **Distribution fitting**                                                                               |
| `pydrght.MSDI`           | **Multivariate Standardized Drought Index** (MSDI)                                                              |
| `pydrght.PNI`            | **Percent of Normal Index** (PNI)                                                                               |
| `pydrght.RAI`            | **Rainfall Anomaly Index** (RAI)                                                                                |
| `pydrght.RDI`            | **Reconnaissance Drought Index** (RDI)                                                                          |
| `pydrght.SI`             | **Standardized Index class** for SPI, SPEI, SSFI, SGI, SSMI                                                     |
| `pydrght.TSDI`             | **Trivariate Standardized Drought Index**                                                      |
| `pydrght.copulas`        | Copula classes: `ClaytonCopula`, `FrankCopula`, `GumbelCopula`, `GaussianCopula`, `GalambosCopula`, `PlackettCopula` |
| `pydrght.pet`            | Potential Evapotranspiration (PET) methods: `hargreaves`,  `thornthwaite`                      |
| `pydrght.methods`            | Unbiased-PWM L-moment estimators and CDF functions for: `Gamma`,  `Generalized Logistic` and, `Pearson III`                       |
| `pydrght.indices`            | Standardized index subclasses using default probability distributions: `SPI`,  `SPEI`, `SGI`, `SSFI` and, `SSMI`                       |
| `pydrght.utils`          | Utility functions: `uni_emp`, `multi_emp`, `accu`, `tri_emp`,`plot_index_with_severity`                                                               |
| `pydrght/examples`       | Example datasets (`.csv`) and usage notebooks (`.ipynb`)                                                         |

---

## Examples & Data

Check the [`examples/`](examples) folder for:

- Sample datasets (`data.csv`, `spi.csv`, `dchar.csv`)  
- Jupyter notebooks (`example_SI.ipynb`, `example_MSDI.ipynb`, etc.)  

---

## References

- McKee, T. B., Doesken, N. J., & Kleist, J. (1993). *The relationship of drought frequency and duration to time scales*. Proceedings of the 8th Conference on Applied Climatology, 179–184.

- Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). *A multiscalar drought index sensitive to global warming: The Standardized Precipitation Evapotranspiration Index*. Journal of Climate, 23(7), 1696–1718. [DOI: 10.1175/2009JCLI2909.1](https://doi.org/10.1175/2009JCLI2909.1)

- Shukla, S., & Wood, A. W. (2008). *Use of a standardized runoff index for characterizing hydrologic drought*. Geophysical Research Letters, 35(2), L02405. [DOI: 10.1029/2007GL032487](https://doi.org/10.1029/2007GL032487)

- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges*. Publications de l'Institut de Statistique de l'Université de Paris, 8, 229–231. [DOI: 10.2139/ssrn.4198458](https://doi.org/10.2139/ssrn.4198458)

- Gibbs, W. J., & Maher, J. V. (1967). *Rainfall deciles as drought indicators*. Bureau of Meteorology, Australia.

- Hayes, M. J. (1999). *Drought Indices*. National Drought Mitigation Center, University of Nebraska-Lincoln.

- van Rooy, M. P. (1965). *A rainfall anomaly index (RAI) independent of time and space*. Notos, 14, 43–48.

- Hänsel, S., Schucknecht, A., & Matschullat, J. (2016). *The Modified Rainfall Anomaly Index (mRAI)—is this an alternative to the Standardised Precipitation Index (SPI) in evaluating future extreme precipitation characteristics?* Theoretical and Applied Climatology, 123, 827–844. [DOI: 10.1007/s00704-015-1389-y](https://doi.org/10.1007/s00704-015-1389-y)

- Tsakiris, G., & Vangelis, H. (2005). *Establishing a drought index incorporating evapotranspiration*. European Water, 9/10, 3–11.

- Hayes, M., Svoboda, M., Wall, N., & Widhalm, M. (2011). *The Lincoln Declaration on Drought Indices: Universal Meteorological Drought Index Recommended*. Bulletin of the American Meteorological Society, 92(4), 485–488. [DOI: 10.1175/2010BAMS3103.1](https://doi.org/10.1175/2010BAMS3103.1)

- Farahmand, A., & AghaKouchak, A. (2015). *A generalized framework for deriving nonparametric standardized drought indicators*. Advances in Water Resources, 76, 140–145. [DOI: 10.1016/j.advwatres.2014.11.012](https://doi.org/10.1016/j.advwatres.2014.11.012)

- Hao, Z., & AghaKouchak, A. (2013). *Multivariate standardized drought index: a parametric multi-index model*. Advances in Water Resources, 57, 12–18. [DOI: 10.1016/j.advwatres.2013.03.009](https://doi.org/10.1016/j.advwatres.2013.03.009)

- Hao, Z., & AghaKouchak, A. (2014). *A nonparametric multivariate multi-index drought monitoring framework*. Journal of Hydrometeorology, 15(1), 89–101. [DOI: 10.1175/jhm-d-12-0160.1](https://doi.org/10.1175/jhm-d-12-0160.1)

- Wu, H., Hayes, M. J., Weiss, A., & Hu, Q. (2001). *An evaluation of the Standardized Precipitation Index, the China-Z Index and the statistical Z-Score*. International Journal of Climatology, 21(6), 745–758. [DOI: 10.1002/joc.658](https://doi.org/10.1002/joc.658)

---

## License

PyDRGHT is licensed under the MIT License – see the [LICENSE](LICENSE) file for details. Please cite the package if you use it in your work.

---

## Citation

If you use **PyDRGHT** in your research, please cite the following article:

> Terzi, T.B., 2026. PyDRGHT: A comprehensive python package for drought analysis. Environmental Modelling & Software 197, 106847. https://doi.org/10.1016/j.envsoft.2025.106847


---
