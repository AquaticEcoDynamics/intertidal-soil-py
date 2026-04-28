"""Atmospheric helper functions translated from aed_land.F90.

Includes WETAIR, VAPPRS, and potential evaporation.
"""

import numpy as np


def vapprs(db: float) -> float:
    """Saturation vapour pressure (Pa) from dry-bulb temperature (degC).

    Translated from VAPPRS in aed_land.F90 (NicheMapR / Smithsonian tables).
    """
    db = np.clip(db, -40.0, 100.0)
    t = db + 273.16
    if t > 273.16:
        loge = (
            -7.90298 * (373.16 / t - 1.0)
            + 5.02808 * np.log10(373.16 / t)
            - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / 373.16)) - 1.0)
            + 8.1328e-3 * (10.0 ** (-3.49149 * (373.16 / t - 1.0)) - 1.0)
            + np.log10(1013.246)
        )
    else:
        loge = (
            -9.09718 * (273.16 / t - 1.0)
            - 3.56654 * np.log10(273.16 / t)
            + 0.876793 * (1.0 - t / 273.16)
            + np.log10(6.1071)
        )
    return (10.0 ** loge) * 100.0


def wetair(
    db: float,
    bp: float = 101325.0,
    rh: float = -1.0,
    wb: float = 0.0,
    dp: float = 999.0,
):
    """Properties of humid air.

    Translated from WETAIR in aed_land.F90 (NicheMapR, Kearney & Porter 2018).

    Parameters
    ----------
    db : float  — dry-bulb temperature (degC)
    bp : float  — barometric pressure (Pa)
    rh : float  — relative humidity (%), set -1 if using wb
    wb : float  — wet-bulb temperature (degC), set 0 if using rh
    dp : float  — dew-point temperature (degC), set 999 if using rh/wb

    Returns
    -------
    dict with keys: e, esat, vd, rw, tvir, tvinc, denair, cp, wtrpot
    """
    tk = db + 273.15
    esat = vapprs(db)

    if dp < 999.0:
        e = vapprs(dp)
        rh = (e / esat) * 100.0
    elif rh > -1.0:
        e = esat * rh / 100.0
    else:
        wbd = db - wb
        wbsat = vapprs(wb)
        dltae = 0.000660 * (1.0 + 0.00115 * wb) * bp * wbd
        e = wbsat - dltae
        rh = (e / esat) * 100.0

    rw = (0.62197 * 1.0053 * e) / (bp - 1.0053 * e)
    vd = e * 0.018016 / (0.998 * 8.31434 * tk)
    tvir = tk * ((1.0 + rw / (18.016 / 28.966)) / (1.0 + rw))
    tvinc = tvir - tk
    denair = 0.0034838 * bp / (0.999 * tvir)
    cp = (1004.84 + rw * 1846.40) / (1.0 + rw)
    wtrpot = 4.615e5 * tk * np.log(rh / 100.0) if rh > 0.0 else -999.0

    return dict(
        e=e, esat=esat, vd=vd, rw=rw, tvir=tvir,
        tvinc=tvinc, denair=denair, cp=cp, wtrpot=wtrpot,
    )


def potential_evaporation(rain: float, air_temp: float) -> float:
    """Simple potential evaporation (m/day).

    Translated from GetPotlEvap in aed_land.F90.
    Returns 0 when rain >= 5 mm/day.
    """
    if rain < 0.005:
        return air_temp / 3500.0
    return 0.0
