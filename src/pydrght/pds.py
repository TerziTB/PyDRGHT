import pandas as pd
import numpy as np

class PDS:
    """
    Class to calculate the Probability of Drought Severity (PDS).
    """
    def __init__(self, data, bins=None, labels=None):
        if hasattr(data, "to_series"):
            data = data.to_series()
        
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        self.data = data.dropna()
        
        self.bins = bins if bins is not None else [-np.inf, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, np.inf]
        self.labels = labels if labels is not None else [
            "Extremely dry", "Severely dry", "Moderately dry", 
            "Near normal", "Moderately wet", "Very wet", "Extremely wet"
        ]
        self.results = None

    def calculate(self):
        """Categorizes data and calculates the probability of each severity class."""
        classes = pd.cut(self.data, bins=self.bins, labels=self.labels)
        
        counts = classes.value_counts().reindex(self.labels, fill_value=0)
        probabilities = counts / len(self.data)
        
        # Create result table
        self.results = pd.DataFrame({
            'Count': counts,
            'Probability': probabilities,
            'Percentage (%)': (probabilities * 100).round(2)
        })
        return self.results

    def total_drought_prob(self):
        """Sums the probability of all dry categories (<= -1.0)."""
        if self.results is None: self.calculate()
        dry_cats = ["Extremely dry", "Severely dry", "Moderately dry"]
        return self.results.loc[dry_cats, 'Probability'].sum()
    