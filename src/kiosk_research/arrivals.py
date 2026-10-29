"""Daily visitor counts and hourly arrival shapes restored from the empirical notebook."""
import random, tempfile
from functools import lru_cache
from pathlib import Path
from kiosk_sim.arrival import load_sales_profiles
from kiosk_sim.visitor_distribution import daily_visitors

def visitor_count(config,case,seed):
    mean=config.visitors if config.visitors is not None else float(case['features']['visitor_mean'])
    if config.visitor_variation=='poisson':return daily_visitors(float(mean),random.Random(seed)) if mean>0 else 0
    return round(float(mean))

def start_hour(config,hours):
    # The sales file only covers one calendar day, so long opening hours start earlier.
    return min(config.arrival_start_hour,24-hours) if config.sales_profile_path else config.arrival_start_hour

@lru_cache(maxsize=None)
def _utf8_copy(path):
    # Seoul open-data CSVs are often CP949; the original loader reads UTF-8 only.
    raw=Path(path).read_bytes()
    try:raw.decode('utf-8-sig');return path
    except UnicodeDecodeError:pass
    target=Path(tempfile.gettempdir())/('kiosk_sales_utf8_'+Path(path).name)
    target.write_text(raw.decode('cp949'),encoding='utf-8')
    return str(target)

@lru_cache(maxsize=None)
def _profiles(path,start,hours,quarter):
    path=_utf8_copy(path) if not path.lower().endswith('.zip') else path
    return load_sales_profiles(path,start_hour=start,hours=hours,quarter=quarter or None)

def purpose_profiles(config,hours):
    if not config.sales_profile_path:return None
    return _profiles(config.sales_profile_path,start_hour(config,hours),hours,config.sales_quarter)
