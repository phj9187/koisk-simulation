"""Compare frozen simulation outputs with held-out daily observations; never fit."""
import argparse,json
from pathlib import Path
import pandas as pd
KEYS=['date','store_id']
METRICS=['visitors','orders','mean_total_wait_minutes','expired_units','net_profit']
def compare(observed,simulated):
 if observed.empty or simulated.empty:raise ValueError('empty validation data')
 observed=observed.copy();simulated=simulated.copy()
 for name,frame in [('observed',observed),('simulated',simulated)]:
  if set(KEYS+METRICS)-set(frame):raise ValueError(name+' missing required columns')
  if frame.duplicated(KEYS).any():raise ValueError(name+' has duplicate store dates')
  pd.to_datetime(frame.date,errors='raise')
  for key in METRICS:
   values=pd.to_numeric(frame[key],errors='raise')
   if values.isna().any() or not __import__('numpy').isfinite(values).all():raise ValueError('non-finite '+key)
   frame[key]=values
   if key!='net_profit' and (values<0).any():raise ValueError('negative '+key)
  if (frame.orders>frame.visitors).any():raise ValueError('orders exceed visitors')
 joined=observed.merge(simulated,on=KEYS,how='outer',suffixes=('_observed','_simulated'),indicator=True,validate='one_to_one')
 if not joined._merge.eq('both').all():raise ValueError('unmatched store/date; no silent dropping')
 return dict(matched_store_dates=len(joined),scope='held-out comparison only; these errors are not calibration parameters',metrics={key:dict(mean_observed=float(joined[key+'_observed'].mean()),mean_simulated=float(joined[key+'_simulated'].mean()),mae=float((joined[key+'_simulated']-joined[key+'_observed']).abs().mean()),bias=float((joined[key+'_simulated']-joined[key+'_observed']).mean())) for key in METRICS})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--observed',required=True);p.add_argument('--simulated',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 result=compare(pd.read_csv(a.observed),pd.read_csv(a.simulated));Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
