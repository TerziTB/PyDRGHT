import calendar
import math
import numpy as np
import pandas as pd

from .utils import solar_declination, month_lengths, mid_month_days, tan_lat_rad

def thornthwaite(start_date, monthly_tmean, latitude):
    """
    Thornthwaite Potential Evapotranspiration (PET) for monthly temperature data.

    Calculates monthly potential evapotranspiration using the Thornthwaite method
    (Thornthwaite, 1948) with day length correction. This method estimates PET from
    monthly mean temperatures and latitude, suitable for long-term drought and water
    balance studies.

    Parameters
    ----------
    monthly_tmean : array-like
        Monthly mean temperature values in degrees Celsius. Negative temperatures 
        are set to 0 for calculation purposes.
    latitude : float
        Latitude of the location in decimal degrees (positive for Northern Hemisphere, 
        negative for Southern Hemisphere).
    start_date : str
        Start date of the series in 'YYYY-MM' format. Used to compute month lengths 
        and mid-month days for solar declination.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the following columns:
        - 'Year': Year of observation
        - 'Month': Month abbreviation (Jan, Feb, …)
        - 'Tmean': Monthly mean temperature (°C)
        - 'MonthlyPET': Thornthwaite PET estimate (mm/month)

    References
    ----------
    - Thornthwaite, C. W. (1948). 
    *An Approach Toward a Rational Classification of Climate*. 
    Geographical Review, 38(1), 55–94.
    """
    start_year, start_month = map(int, start_date.split('-'))
    data_length = len(monthly_tmean)

    mlen_array = month_lengths(start_year, start_month, data_length)

    msum = mid_month_days()

    tan_lat = tan_lat_rad(latitude)

    month_cycle = np.array([(start_month + i - 1) % 12 + 1 for i in range(data_length)])

    monthly_means = np.array([max(0, np.mean([t for t, m in zip(monthly_tmean, month_cycle) if m == month]))
                              for month in range(1, 13)])
    
    heat_index = np.sum((monthly_means / 5) ** 1.514) or 0.001
    J2, J3 = heat_index**2, heat_index**3
    alpha = 0.000000675 * J3 - 0.0000771 * J2 + 0.01792 * heat_index + 0.49239

    total_months = start_month + data_length - 1
    end_year = start_year + total_months // 12
    end_month = total_months % 12 or 12
    if end_month == 12: end_year -= 1 if total_months % 12 == 0 else 0

    results = []
    month_count = 0

    for year in range(start_year, end_year + 1):
        month_start = start_month if year == start_year else 1
        month_end = end_month if year == end_year else 12

        for month in range(month_start, month_end + 1):
            if month_count >= data_length:
                break

            tmean = monthly_tmean.iloc[month_count]
            if tmean < 0:
                pet = 0
            else:
                delta = solar_declination(msum[month - 1])

                omega = math.acos(max(-1, min(1, -tan_lat * math.tan(delta))))
                N = 24 / math.pi * omega

                K = N / 12 * mlen_array[month_count] / 30

                pet = K * 16 * ((10 * tmean / heat_index) ** alpha)

            results.append({
                'Year': year,
                'Month': calendar.month_abbr[month],
                'Tmean': round(tmean, 3),
                'MonthlyPET': round(pet, 3)
            })

            month_count += 1

    return pd.DataFrame(results)
