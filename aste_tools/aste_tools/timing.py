"""Time-step helpers for MITgcm/ASTE output."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def is_leap(year: int) -> bool:
    """Return True when ``year`` is a Gregorian leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def timestep_to_datetime(ts: int | float, *, delta_t: int = 1200, start: datetime = datetime(1992, 1, 1)) -> datetime:
    """Convert a model time step to a Python datetime."""

    return start + timedelta(seconds=float(ts) * delta_t)


def monthly_file_steps(delta_t: int, start_year: int, end_year: int) -> np.ndarray:
    """Return cumulative month-end time steps from ``start_year`` through ``end_year - 1``."""

    days_regular = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_leap = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    steps = []
    for year in range(start_year, end_year):
        days = days_leap if is_leap(year) else days_regular
        steps.extend(days * 24 * 3600 / delta_t)
    return np.cumsum(np.asarray(steps, dtype=int))


def select_month_steps(
    months_by_year: dict[str | int, np.ndarray | list[int]],
    file_steps: np.ndarray,
    *,
    delta_t: int,
    start: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Select zero-padded step strings and datetimes for requested months."""

    step_strings: list[str] = []
    datetimes: list[datetime] = []
    requested = {int(year): set(int(month) for month in months) for year, months in months_by_year.items()}
    for step in file_steps:
        dte = timestep_to_datetime(int(step), delta_t=delta_t, start=start)
        if dte.year in requested and dte.month in requested[dte.year]:
            datetimes.append(dte)
            step_strings.append(str(int(step)).zfill(10))
    return np.asarray(step_strings), np.asarray(datetimes, dtype=object)

