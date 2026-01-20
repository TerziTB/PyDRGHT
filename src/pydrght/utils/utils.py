import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def uni_emp(X, method='Gringorten') -> pd.Series:
    """
    Compute the univariate empirical cumulative distribution function (CDF).

    Parameters
    ----------
    X : pd.Series or array-like
        1D input data (e.g., precipitation, temperature).
    method : str, default='Gringorten'
        Plotting position formula to use. Options:
        - 'Gringorten' : recommended for hydrology
        - 'Weibull'    : standard Weibull formula

    Returns
    -------
    pd.Series
        Empirical CDF values corresponding to input data, preserving the original index.

    Notes
    -----
    Empirical CDF is computed by ranking each value and applying the plotting position formula.

    References
    ----------
    Gringorten, I. I. (1963). A plotting rule for extreme probability paper.
    Journal of Geophysical Research, 68(3), 813–814.

    Weibull, W. (1939). A statistical distribution function of wide applicability.
    Journal of Applied Mechanics, 18, 293–297.
    """
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    
    X = X.dropna()
    n = len(X)
    
    S = X.apply(lambda xi: (X <= xi).sum())

    if method == 'Gringorten':
        cdf = (S - 0.44) / (n + 0.12)
    elif method == 'Blom':
        cdf = (S - 0.375) / (n + 0.25)
    elif method == 'Hazen':
        cdf = (S - 0.50) / n
    elif method == 'California':
        cdf = (S - 1) / n
    elif method == 'Tukey':
        cdf = (3 * S - 1) / (3 * n + 1)
    elif method == 'Weibull':
        cdf = S / (n + 1)
    else:
        raise ValueError("Method must be 'Gringorten', 'Blom', 'Cunnane', 'Hazen', 'Weibull', 'California', or 'Tukey'")

    return pd.Series(cdf.values, index=X.index)

def accu(X, ts):
    """
    Compute accumulated or averaged values over a specified time scale.

    Parameters
    ----------
    X : pd.Series or array-like
        Input 1D data (e.g., monthly precipitation or temperature).
    ts : int
        Time scale for accumulation (e.g., 3 for a 3-month accumulation).

    Returns
    -------
    pd.Series
        Accumulated/averaged values with NaNs for initial positions where insufficient data exists.

    Notes
    -----
    This function uses rolling slices to compute the mean of `ts` consecutive values.
    Useful for creating time-scaled indices such as SPI or SPEI.
    """
    if isinstance(X, pd.Series):
        index = X.index
        X_values = X.values
    else:
        X_values = np.asarray(X).flatten()
        index = pd.RangeIndex(len(X_values))

    if ts < 1:
        raise ValueError("Time scale (ts) must be a positive integer.")
    if len(X_values) < ts:
        raise ValueError("Length of input data must be >= time scale.")

    slices = [X_values[i:len(X_values) - ts + i + 1] for i in range(ts)]
    stacked = np.stack(slices, axis=1)

    averaged = stacked.mean(axis=1)

    valid_index = index[ts-1:]

    return pd.Series(averaged, index=valid_index, name=getattr(X, 'name', None))

def multi_emp(X, Y, method='Gringorten') -> pd.Series:
    """
    Compute joint empirical probabilities for bivariate data using a plotting position formula.

    Parameters
    ----------
    X : pd.Series or array-like
        First variable.
    Y : pd.Series or array-like
        Second variable.
    method : str, default='Gringorten'
        Plotting position formula. Options: 'Gringorten', 'Weibull'.

    Returns
    -------
    pd.Series
        Joint empirical probabilities, indexed like X.
        
    Notes
    -----
    Joint empirical probability is calculated as the proportion of observations
    less than or equal to both X_i and Y_i. Useful for multivariate analyses.
    """
    X = pd.Series(X) if not isinstance(X, pd.Series) else X
    Y = pd.Series(Y) if not isinstance(Y, pd.Series) else Y

    df = pd.concat([X, Y], axis=1).dropna()
    X, Y = df.iloc[:, 0], df.iloc[:, 1]

    n = len(X)
    S = np.empty(n)

    for k in range(n):
        count = np.sum((X <= X.iloc[k]) & (Y <= Y.iloc[k]))
        if method == 'Gringorten':
            S[k] = (count - 0.44) / (n + 0.12)
        elif method == 'Blom':
            S[k] = (count - 0.375) / (n + 0.25)
        elif method == 'Cunnane':
            S[k] = (count - 0.40) / (n + 0.20)
        elif method == 'Hazen':
            S[k] = (count - 0.50) / n
        elif method == 'Weibull':
            S[k] = count / (n + 1)
        elif method == 'California':
            S[k] = (count - 1) / n
        elif method == 'Tukey':
            S[k] = (3 * count - 1) / (3 * n + 1)
        else:
            raise ValueError("Method must be 'Gringorten', 'Blom', 'Cunnane', 'Hazen', 'Weibull', 'California', or 'Tukey'")

    return pd.Series(S, index=X.index)

def tri_emp(X, Y, Z, method='Gringorten') -> pd.Series:
    """
    Compute joint empirical probabilities for trivariate data using a plotting position formula.

    Parameters
    ----------
    X, Y, Z : pd.Series or array-like
        Input variables.
    method : str, default='Gringorten'
        Plotting position formula. Options: 'Gringorten', 'Weibull'.

    Returns
    -------
    pd.Series
        Joint empirical probabilities, indexed like X.

    Notes
    -----
    The joint empirical probability is the proportion of observations
    less than or equal to X_i, Y_i, and Z_i for each i. 
    """
    X = pd.Series(X, name="X")
    Y = pd.Series(Y, name="Y")
    Z = pd.Series(Z, name="Z")

    df = pd.concat([X, Y, Z], axis=1).dropna()
    X, Y, Z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

    n = len(X)
    S = np.empty(n)

    for k in range(n):
        count = np.sum((X <= X.iloc[k]) & (Y <= Y.iloc[k]) & (Z <= Z.iloc[k]))
        if method == 'Gringorten':
            S[k] = (count - 0.44) / (n + 0.12)
        elif method == 'Blom':
            S[k] = (count - 0.375) / (n + 0.25)
        elif method == 'Cunnane':
            S[k] = (count - 0.40) / (n + 0.20)
        elif method == 'Hazen':
            S[k] = (count - 0.50) / n
        elif method == 'Weibull':
            S[k] = count / (n + 1)
        elif method == 'California':
            S[k] = (count - 1) / n
        elif method == 'Tukey':
            S[k] = (3 * count - 1) / (3 * n + 1)
        else:
            raise ValueError("Method must be 'Gringorten', 'Blom', 'Cunnane', 'Hazen', 'Weibull', 'California', or 'Tukey'")

    return pd.Series(S, index=X.index)

def plot_index(*indices):
    """
    Plot one or more standardized indices from pandas Series.

    Parameters
    ----------
    *indices : pd.Series
        One or more pandas Series to plot. The index should be datetime-like or numeric.
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(12, 5))

    for i, series in enumerate(indices):
        label = series.name if series.name is not None else f"Index_{i+1}"
        plt.plot(series.index, series.values, label=label, color=colors[i % len(colors)], linewidth=1.5)

    plt.axhline(0, color="black", linestyle="-", linewidth=1.5)

    plt.ylim(-4, 4)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=True)

    plt.legend(loc='upper left')

    plt.xlabel("Time (Month)")
    plt.ylabel("Index Value")
    plt.show()


def plot_index_with_severity(*series_list):
    """
    Plot standardized indices with areas below 0 filled in red.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing indices to plot.
    indices : list
        Column names of the indices to plot.
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]  # extend if needed

    plt.figure(figsize=(12, 5))
    
    for i, s in enumerate(series_list):
        x = s.index
        y = s.values
        plt.plot(x, y, color=colors[i % len(colors)], linewidth=1.5, label="Index")
        
        plt.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.3, label="< 0")
        plt.fill_between(x, y, 0, where=(y < -1), color='red', alpha=0.6, label="< -1")
        plt.fill_between(x, y, 0, where=(y < -2), color='darkred', alpha=0.9, label="< -2") 

    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.axhline(-1, color="red", linestyle="--", linewidth=1.5)
    plt.axhline(-2, color="red", linestyle="--", linewidth=1.5)
    
    plt.ylim(-4, 4)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)  # thicker left spine

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=True)

    plt.legend(loc='upper left')

    plt.xlabel("Time (Month)")
    plt.ylabel("Index Value")
    plt.show()