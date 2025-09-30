import calendar
import pandas as pd

from .utils import solar_declination, month_lengths, mid_month_days, inverse_rel_dist, extraterrestrial_radiation, sunset_hour_angle

def hargreaves_pet(tmin, tmax, et_rad, tmean=None):
    """
    Hargreaves reference evapotranspiration (mm/day)
    """
    if tmean is None:
        tmean = (tmin + tmax) / 2
    trange = max(0, tmax - tmin)
    return 0.0023 * 0.408 * et_rad * (tmean + 17.8) * trange**0.5

def hargreaves(start_date, tmin, tmax, latitude, tmean=None):
    """
    Calculate monthly Hargreaves PET for a time series of min/max temperatures.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM' format.
    tmin : array-like
        Monthly minimum temperatures (°C)
    tmax : array-like
        Monthly maximum temperatures (°C)
    latitude : float
        Latitude in decimal degrees
    tmean : array-like, optional
        Monthly mean temperatures (°C). If None, calculated as (tmin+tmax)/2.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly PET and intermediate calculations.

    References
    ----------
    - Hargreaves, G. H., & Samani, Z. A. (1985). 
    *Reference crop evapotranspiration from temperature*. 
    Applied Engineering in Agriculture, 1(2), 96–99.

    - Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). 
    *Crop evapotranspiration – Guidelines for computing crop water requirements*. 
    FAO Irrigation and Drainage Paper 56.
    """
    start_year, start_month = map(int, start_date.split('-'))
    n_months = len(tmin)

    mlen_array = month_lengths(start_year, start_month, n_months)
    msum = mid_month_days()
    latitude_rad = latitude / 57.2957795
    
    results = []
    month_count = 0
    total_months = start_month + n_months - 1
    end_year = start_year + total_months // 12
    end_month = total_months % 12 or 12

    for year in range(start_year, end_year + 1):
        month_start = start_month if year == start_year else 1
        month_end = end_month if year == end_year else 12

        for month in range(month_start, month_end + 1):
            if month_count >= n_months:
                break

            t_min = tmin.iloc[month_count]
            t_max = tmax.iloc[month_count]
            t_m = tmean.iloc[month_count] if tmean is not None else None

            doy = msum[month - 1]

            sol_dec = solar_declination(doy)
            sha = sunset_hour_angle(latitude_rad, sol_dec)
            ird = inverse_rel_dist(doy)
            ra = extraterrestrial_radiation(latitude_rad, sol_dec, sha, ird)

            daily_pet = hargreaves_pet(t_min, t_max, ra, t_m)

            monthly_pet = daily_pet * mlen_array[month_count]

            results.append({
                'Year': year,
                'Month': calendar.month_abbr[month],
                'Tmin': round(t_min, 3),
                'Tmax': round(t_max, 3),
                'Tmean': round(t_m if t_m is not None else (t_min + t_max)/2, 3),
                'Ra': round(ra, 3),
                'MonthlyPET': round(monthly_pet, 3)
            })

            month_count += 1

    return pd.DataFrame(results)