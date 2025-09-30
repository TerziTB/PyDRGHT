import calendar
import math
import numpy as np

def monthly_mean_daylen(latitude, month):
    """
    Calculate the mean day length (hours) for a given month and latitude.
    
    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    month : int
        Month number (1-12).
    
    Returns
    -------
    float
        Mean day length in hours.
    """
    lat_rad = latitude * math.pi / 180.0
    mlen = [31,28,31,30,31,30,31,31,30,31,30,31]
    cumsum = np.cumsum(mlen)
    msum = [cs - m + 15 for cs, m in zip(cumsum, mlen)]
    j = msum[month-1]
    delta = 0.409 * math.sin(0.0172 * j - 1.39)
    tmp = -math.tan(lat_rad) * math.tan(delta)
    if tmp <= -1: return 24.0
    elif tmp >= 1: return 0.0
    else: return 24.0 * math.acos(tmp) / math.pi

def solar_declination(j):
    """
    Calculate solar declination for a given day of the year.

    Parameters
    ----------
    j : int
        Day of the year (1-365)

    Returns
    -------
    float
        Solar declination in radians.
    """
    return 0.4093 * math.sin((2 * math.pi * j / 365) - 1.405)

def month_lengths(start_year, start_month, n_months):
    """
    Generate a list of month lengths for a given period.

    Parameters
    ----------
    start_year : int
        Starting year
    start_month : int
        Starting month (1-12)
    n_months : int
        Number of months

    Returns
    -------
    list of int
        Number of days in each month for the period.
    """
    lengths = []
    year, month = start_year, start_month
    for _ in range(n_months):
        lengths.append(calendar.monthrange(year, month)[1])
        month += 1
        if month > 12:
            month = 1
            year += 1
    return lengths

def mid_month_days():
    """
    Calculate the day of year corresponding to the middle of each month (non-leap year).

    Returns
    -------
    list of int
        Mid-month day numbers (1-365).
    """
    mlen = [31,28,31,30,31,30,31,31,30,31,30,31]
    cumsum = np.cumsum(mlen)
    return [cs - m + 15 for cs, m in zip(cumsum, mlen)]

def tan_lat_rad(latitude):
    """
    Calculate tangent of latitude in radians.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.

    Returns
    -------
    float
        Tangent of latitude in radians.
    """
    return math.tan(latitude / 57.2957795)

def inverse_rel_dist(day):
    """
    Inverse relative distance Earth-Sun (FAO Eq. 25).

    Parameters
    ----------
    day : int
        Day of the year (1-365)

    Returns
    -------
    float
        Inverse relative distance.
    """
    return 1 + 0.033 * math.cos(0.0172 * day)

def extraterrestrial_radiation(latitude, sol_dec, sha, ird):
    """
    Extraterrestrial radiation (Ra, mm/day) following FAO 56 / SPEI package.

    Parameters
    ----------
    latitude : float
        Latitude in radians.
    sol_dec : float
        Solar declination in radians.
    sha : float
        Sunset hour angle in radians.
    ird : float
        Inverse relative distance Earth-Sun.

    Returns
    -------
    float
        Extraterrestrial radiation (Ra) in mm/day.
    """
    ra = 37.6 * ird * (sha * math.sin(latitude) * math.sin(sol_dec) +
                       math.cos(latitude) * math.cos(sol_dec) * math.sin(sha))
    return max(0, ra)

def sunset_hour_angle(latitude, solar_dec):
    """
    Calculate sunset hour angle (radians).

    Parameters
    ----------
    latitude : float
        Latitude in radians.
    solar_dec : float
        Solar declination in radians.

    Returns
    -------
    float
        Sunset hour angle in radians.
    """
    sset = -math.tan(latitude) * math.tan(solar_dec)
    if abs(sset) <= 1:
        return math.acos(sset)
    elif sset < -1:
        return math.pi
    else:
        return 0.0
