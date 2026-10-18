"""Load hourly population counts; shape is a proxy, not restaurant footfall."""
import csv
import io
import zipfile
from contextlib import contextmanager
import re
from datetime import datetime
from collections import defaultdict

HOURS = ("시간대구분", "시간대", "기준시간", "hour")
COUNTS = ("총생활인구수", "총생활인구", "생활인구수", "유동인구수", "count")
AREAS = ("행정동코드", "행정동명", "상권코드", "상권명", "area")


def first(fields, names):
    return next((x for x in names if x in fields), None)


def parse_hour(raw):
    match = re.match(r"^(\d{1,2})(?::\d\d)?$", raw.strip())
    if not match or int(match.group(1)) > 23:
        raise ValueError(f"Unsupported hour {raw!r}; expected 0..23 or HH:MM")
    return int(match.group(1))


@contextmanager
def population_file(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("ZIP must contain exactly one CSV")
            with archive.open(names[0]) as data:
                with io.TextIOWrapper(data, encoding="utf-8-sig", newline="") as file:
                    yield file
    else:
        with open(path, encoding="utf-8-sig", newline="") as file:
            yield file


def load_hourly_weights(path, *, area=None, start_hour=11, hours=8, day_type=None):
    """Normalize the monthly mean count at each hour across operating hours."""
    if not (0 <= start_hour <= 23 and 1 <= hours <= 24 - start_hour):
        raise ValueError("Choose operating hours within one calendar day")
    with population_file(path) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        hour_field, count_field, area_field = (first(fields, x) for x in (HOURS, COUNTS, AREAS))
        if not hour_field or not count_field:
            raise ValueError(f"Required hour and population count columns missing. Found: {fields}")
        if area is not None and not area_field:
            raise ValueError("Area filter requested but no area column found")
        if day_type not in (None, "weekday", "weekend"):
            raise ValueError("day_type must be weekday or weekend")
        if day_type and "기준일ID" not in fields:
            raise ValueError("Weekday split requires 기준일ID")
        totals = defaultdict(float)
        for row in reader:
            if area is not None and row[area_field].strip() != str(area):
                continue
            if day_type:
                weekend = datetime.strptime(row["기준일ID"], "%Y%m%d").weekday() >= 5
                if weekend != (day_type == "weekend"):
                    continue
            hour = parse_hour(row[hour_field])
            if start_hour <= hour < start_hour + hours:
                count = float(row[count_field].replace(",", ""))
                if count < 0:
                    raise ValueError("Population counts must be nonnegative")
                totals[hour] += count
    expected = range(start_hour, start_hour + hours)
    if any(hour not in totals for hour in expected) or sum(totals.values()) <= 0:
        raise ValueError("No complete positive hourly profile for selected area and hours")
    # Each hour has the same date coverage in the official monthly file.
    denom = sum(totals.values())
    return tuple(totals[hour] / denom for hour in expected)
