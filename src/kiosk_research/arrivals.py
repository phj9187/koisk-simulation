"""Daily visitor counts and hourly arrival shapes restored from the empirical notebook."""
import random, tempfile, hashlib, csv
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
    return min(config.arrival_start_hour,24-hours) if config.sales_profile_path or config.realism.get('population_path') else config.arrival_start_hour

@lru_cache(maxsize=None)
def _utf8_copy(path,signature=None):
    # Seoul open-data CSVs are often CP949; the original loader reads UTF-8 only.
    raw=Path(path).read_bytes()
    try:raw.decode('utf-8-sig');return path
    except UnicodeDecodeError:pass
    target=Path(tempfile.gettempdir())/('kiosk_sales_utf8_'+hashlib.sha256(raw).hexdigest()+'.csv')
    target.write_text(raw.decode('cp949'),encoding='utf-8')
    return str(target)

@lru_cache(maxsize=None)
def _profiles(path,start,hours,quarter,signature):
    path=_utf8_copy(path,signature) if not path.lower().endswith('.zip') else path
    return load_sales_profiles(path,start_hour=start,hours=hours,quarter=quarter or None)

def signature(path):
    stat=Path(path).stat();return stat.st_mtime_ns,stat.st_size


@lru_cache(maxsize=32)
def _population_profile(path,area,start,hours,day_type,file_signature):
    from kiosk_sim.public_data import load_hourly_weights
    return load_hourly_weights(_utf8_copy(path,file_signature),area=area or None,start_hour=start,hours=hours,day_type=day_type)


def purpose_profiles(config,hours,calendar=None):
    from kiosk_research.realism import options
    from kiosk_sim.arrival import assumed_purpose_shape,arrival_profile
    from kiosk_sim.public_data import load_hourly_weights
    settings=options(config);start=start_hour(config,hours)
    profiles=_profiles(config.sales_profile_path,start,hours,config.sales_quarter,signature(config.sales_profile_path)) if config.sales_profile_path else None
    if not settings.population_path:return profiles
    population=_population_profile(settings.population_path,settings.population_area,start,hours,
                                   ('weekend' if calendar.weekday()>=5 else 'weekday') if calendar else None,signature(settings.population_path))
    profiles=profiles or {p:assumed_purpose_shape(p,start,hours) for p in ('meal','dessert','mixed')}
    return {p:arrival_profile(population,v,population_strength=settings.population_strength) for p,v in profiles.items()}


@lru_cache(maxsize=16)
def _weekday_counts(path,quarter,signature):
    from kiosk_sim.public_data import population_file
    names=('월요일','화요일','수요일','목요일','금요일','토요일','일요일')
    totals=[0.]*7
    path=_utf8_copy(path,signature) if not path.lower().endswith('.zip') else path
    with population_file(path) as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or []
        columns=[name+'_매출_건수' for name in names]
        if not set(columns)<=set(fields):return None
        for row in reader:
            if quarter and row.get('기준_년분기_코드')!=str(quarter):continue
            industry=row.get('서비스_업종_코드_명','')
            if not any(x in industry for x in ('음식점','식당','분식','치킨','패스트푸드','커피','음료','제과','제빵','디저트')):continue
            for i,key in enumerate(columns):
                value=float((row[key] or '0').replace(',',''))
                import math
                if not math.isfinite(value) or value<0:raise ValueError('invalid weekday sales count')
                totals[i]+=value
    if sum(totals)<=0:raise ValueError('empty weekday sales counts')
    mean=sum(totals)/7
    return tuple(v/mean for v in totals)


def sales_day_multiplier(config,calendar):
    if not config.sales_profile_path:return 1.,'assumed'
    counts=_weekday_counts(config.sales_profile_path,config.sales_quarter,signature(config.sales_profile_path))
    return (counts[calendar.weekday()],'sales_counts_separable') if counts else (1.,'assumed')
