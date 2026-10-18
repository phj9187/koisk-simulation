"""Exogenous arrival profiles. Assumed purpose shapes are never claimed as observed data."""
import csv
import math
from collections import defaultdict
from .public_data import population_file

PURPOSES = ("meal", "dessert", "mixed")
BINS = ((0, 6), (6, 11), (11, 14), (14, 17), (17, 21), (21, 24))


def normalize(values):
    total = sum(values)
    if total <= 0 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Arrival weights must be finite, nonnegative and nonzero")
    return tuple(x / total for x in values)


def assumed_purpose_shape(purpose, start_hour=11, hours=8):
    """Illustrative meal/dessert timing hypotheses; tune in sensitivity analysis."""
    def bump(hour, peak, width):
        return math.exp(-.5 * ((hour + .5 - peak) / width) ** 2)
    result = []
    for h in range(start_hour, start_hour + hours):
        meal = .30 + 2.7 * bump(h, 12.5, 1.0) + 2.7 * bump(h, 18.5, 1.1)
        dessert = .45 + 2.1 * bump(h, 15.5, 1.7) + 1.0 * bump(h, 19.5, 1.5)
        result.append(meal if purpose == "meal" else dessert if purpose == "dessert"
                      else (meal + dessert) / 2)
    return normalize(result)


def load_sales_profiles(path, *, start_hour=11, hours=8, quarter=None):
    """Official Seoul estimated sales counts by industry and broad time bins.

    Splitting a 11-14 bin evenly across its hours is an approximation.
    No joint day-of-week x time-of-day breakdown is present in this schema.
    """
    totals = defaultdict(lambda: [0.] * 6)
    with population_file(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        industry = next((x for x in ("서비스_업종_코드_명", "서비스_업종_코드", "industry") if x in fields), None)
        aliases = (
            ("시간대_건수~06_매출_건수", "시간대_00~06_매출_건수", "TMZON_00_06_SELNG_CO"),
            ("시간대_건수~11_매출_건수", "시간대_06~11_매출_건수", "TMZON_06_11_SELNG_CO"),
            ("시간대_건수~14_매출_건수", "시간대_11~14_매출_건수", "TMZON_11_14_SELNG_CO"),
            ("시간대_건수~17_매출_건수", "시간대_14~17_매출_건수", "TMZON_14_17_SELNG_CO"),
            ("시간대_건수~21_매출_건수", "시간대_17~21_매출_건수", "TMZON_17_21_SELNG_CO"),
            ("시간대_건수~24_매출_건수", "시간대_21~24_매출_건수", "TMZON_21_24_SELNG_CO"),
        )
        columns = [next((x for x in group if x in fields), None) for group in aliases]
        if not industry or any(x is None for x in columns):
            raise ValueError("Sales file requires industry and six hourly sales COUNT columns")
        for row in reader:
            if quarter and row.get("기준_년분기_코드") != str(quarter):
                continue
            label = row[industry]
            purpose = ("dessert" if any(x in label for x in ("커피", "음료", "제과", "제빵", "디저트"))
                       else "meal" if any(x in label for x in ("음식점", "식당", "분식", "치킨", "패스트푸드")) else None)
            if purpose:
                for i, column in enumerate(columns):
                    value = float((row[column] or "0").replace(",", ""))
                    if value < 0 or not math.isfinite(value):
                        raise ValueError("Invalid estimated sales count")
                    totals[purpose][i] += value
    if not all(sum(totals[k]) > 0 for k in ("meal", "dessert")):
        raise ValueError("Sales file needs both meal and cafe/bakery industry rows")
    result = {}
    for purpose in ("meal", "dessert"):
        values = []
        for h in range(start_hour, start_hour + hours):
            index = next(i for i, (a, b) in enumerate(BINS) if a <= h < b)
            a, b = BINS[index]
            values.append(totals[purpose][index] / (b - a))
        result[purpose] = normalize(values)
    result["mixed"] = normalize([(a + b) / 2 for a, b in zip(result["meal"], result["dessert"])])
    return result


def arrival_profile(population, purpose_profile, *, population_strength=.3):
    """Geometric blend of normalized proxies, then renormalize.

    The blend strength is a sensitivity parameter, not estimated by the data.
    """
    if not 0 <= population_strength <= 1:
        raise ValueError("population_strength must lie in [0, 1]")
    if population is None:
        return normalize(purpose_profile)
    if len(population) != len(purpose_profile):
        raise ValueError("Hourly profile lengths differ")
    return normalize([max(x, 1e-12) ** population_strength *
                      max(y, 1e-12) ** (1 - population_strength)
                      for x, y in zip(population, purpose_profile)])
