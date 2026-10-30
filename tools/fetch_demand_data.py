"""Download official inputs and retain source URLs, byte hashes and retrieval time.

No credentials, fallback mirrors, or substitution of generated observations.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import requests
import csv
import calendar


def fetch(root):
    from kiosk_cloud.data import ensure_survey
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def save(name,response,url,role):
        response.raise_for_status();data=response.content
        if not data:raise ValueError('empty download')
        target=root/name
        if target.exists() and target.read_bytes()!=data:raise FileExistsError('input changed; choose a new input directory: '+str(target))
        target.write_bytes(data)
        manifest.append({'file':name,'role':role,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()})
        return data
    survey=ensure_survey(root/'source_2025.xlsx')
    from kiosk_sim.empirical_stores import SOURCE_URL
    manifest.append({'file':survey.name,'role':'KREI observed store donors','url':SOURCE_URL,'sha256':hashlib.sha256(survey.read_bytes()).hexdigest(),'bytes':survey.stat().st_size})
    page_url='https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do'
    session=requests.Session();page=session.get(page_url,timeout=60);page.raise_for_status()
    # Select a completed historical year rather than a partial current-year release.
    match=re.search(r'title="[^"]*2025[^\"]*\.zip"[^>]*onclick="javascript:downloadFile\(\x27(\d+)\x27\)',page.text)
    if not match:raise ValueError('official Seoul 2025 download link not found; inspect portal schema')
    seq=match.group(1)
    url='https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false'
    response=session.post(url,data={'infId':'OA-15572','seq':seq,'seqNo':seq,'infSeq':'3'},headers={'Referer':page_url},timeout=120)
    data=save('seoul_sales_2025.zip',response,page_url,'Seoul estimated transaction-count timing and weekday proxy')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError('expected one annual sales CSV')
        raw=archive.read(names[0]);text=None
        for encoding in ('utf-8-sig','cp949'):
            try:text=raw.decode(encoding);break
            except UnicodeDecodeError:pass
        if text is None:raise ValueError('unknown sales encoding')
        target=root/'seoul_sales.csv';target.write_text(text,encoding='utf-8-sig')
        manifest.append({'file':target.name,'derived_from':'seoul_sales_2025.zip','operation':'decode source CSV to UTF-8; observations unchanged','sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
    url='https://www.census.gov/retail/marts/www/adv72200.txt'
    save('us_foodservices_sales_and_factors.txt',session.get(url,timeout=60),url,
         'US food-service monetary sales and adjustment factors; foreign stress scenarios only, not Korean footfall')
    population_url='https://data.seoul.go.kr/dataList/OA-15439/S/1/datasetView.do'
    page=session.get(population_url,timeout=60);page.raise_for_status()
    match=re.search(r'title="LOCAL_PEOPLE_GU_2025\.zip"[^>]*onclick="javascript:downloadFile\(\x27(\d+)\x27\)',page.text)
    if not match:raise ValueError('official 2025 population download missing')
    seq=match.group(1)
    response=session.post('https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false',
                          data={'infId':'OA-15439','seq':seq,'seqNo':seq,'infSeq':'3'},headers={'Referer':population_url},timeout=120)
    data=save('seoul_population_2025.zip',response,population_url,'Seoul hourly population presence proxy; not store footfall')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(names)!=1:raise ValueError('expected one population CSV')
        with archive.open(names[0]) as binary,io.TextIOWrapper(binary,encoding='utf-8-sig') as text:
            reader=csv.DictReader(text)
            with (root/'population_hourly.csv').open('w',encoding='utf-8-sig',newline='') as output:
                writer=csv.writer(output);writer.writerow(['기준일ID','hour','area','count'])
                for row in reader:writer.writerow([row['stdr_de_id'],row['tmzon_pd_se'],row['adstrd_code_se'],row['tot_lvpop_co']])
    path=root/'population_hourly.csv'
    manifest.append({'file':path.name,'derived_from':'seoul_population_2025.zip','operation':'select date, hour, district, total population; rename columns without modifying counts',
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    # Keep one completed historical year. Monthly adjustment factors contain
    # holiday/trading-day effects as well as seasonality, and are a scenario proxy.
    text=(root/'us_foodservices_sales_and_factors.txt').read_text()
    factors=text.split('SEASONAL FACTORS',1)[1]
    line=next(line for line in factors.splitlines() if re.match(r'^2025\s',line.strip()))
    values=[float(v) for v in line.split()[1:]]
    if len(values)!=12:raise ValueError('incomplete historical US factors')
    mean=sum(values)/12
    with (root/'us_2025_monthly_scenario.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['month','multiplier'])
        writer.writerows((i+1,v/mean) for i,v in enumerate(values))
    path=root/'us_2025_monthly_scenario.csv'
    manifest.append({'file':path.name,'derived_from':'us_foodservices_sales_and_factors.txt','operation':'2025 adjustment factors divided by their 12-month mean; foreign stress scenario only',
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    (root/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=True,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='work/realism_inputs')
    fetch(parser.parse_args().root)
