import random
import statistics
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from kiosk_research.arrivals import visitor_count,start_hour,purpose_profiles
from kiosk_research.config import Config
from kiosk_research.engine import Experiment
from kiosk_research.policy import Policy
from kiosk_research.simulation import simulate
from kiosk_research.worker import restore_store,seed_for
from kiosk_sim.generators import customers
from test_pareto_research import small,deterministic

SCENARIO={'customer':[.45,.25,.30],'store':[.75,.25]}
BINS=('00~06','06~11','11~14','14~17','17~21','21~24')


def sales_csv(root,encoding='cp949'):
    # Seoul open-data layout: one row per area and industry, six hourly sales-count bins.
    header=['기준_년분기_코드','서비스_업종_코드_명']+[f'시간대_{b}_매출_건수' for b in BINS]
    rows=[['20251','한식음식점','0','10','900','100','600','50'],
          ['20251','커피-음료','0','80','300','500','200','20'],
          ['20244','한식음식점','0','0','0','0','0','999']]
    path=Path(root)/'seoul_sales.csv'
    path.write_bytes('\n'.join(','.join(r) for r in [header,*rows]).encode(encoding))
    return path


class RealDataInputTests(unittest.TestCase):
    def test_defaults_keep_previous_behaviour(self):
        config=Config().validate()
        case={'features':{'visitor_mean':41.6}}
        self.assertEqual((config.visitor_variation,config.sales_profile_path,config.arrival_start_hour),('fixed','',11))
        self.assertEqual(visitor_count(config,case,1),30)
        self.assertEqual(visitor_count(Config(visitors=None),case,1),42)
        self.assertEqual(start_hour(config,8),11)
        self.assertIsNone(purpose_profiles(config,8))

    def test_actor_critic_presets_use_real_store_data(self):
        import yaml
        for name,stores in (('pilot',8),('research',64),('continual',64)):
            path=Path(__file__).resolve().parents[1]/'config'/f'actor_critic_{name}.yaml'
            config=Config(**yaml.safe_load(path.read_text(encoding='utf-8'))).validate(require_source=False)
            self.assertEqual((config.source,config.stores,config.visitors,config.visitor_variation),('empirical',stores,None,'poisson'))
            self.assertTrue(config.sales_profile_path.endswith('raw_data/seoul_sales.csv'))

    def test_poisson_visitors_are_reproducible_and_vary_by_day(self):
        config=Config(visitors=None,visitor_variation='poisson').validate()
        case={'id':3,'features':{'visitor_mean':56.6}}
        counts=[visitor_count(config,case,seed_for(config.seed,'visitors',day,case['id'])) for day in range(1,401)]
        again=[visitor_count(config,case,seed_for(config.seed,'visitors',day,case['id'])) for day in range(1,401)]
        self.assertEqual(counts,again)
        self.assertGreater(len(set(counts)),10)
        self.assertAlmostEqual(statistics.mean(counts),56.6,delta=2.)
        self.assertEqual(visitor_count(config,{'features':{'visitor_mean':0.}},1),0)

    def test_cp949_seoul_sales_file_becomes_arrival_profile(self):
        with tempfile.TemporaryDirectory() as root:
            path=sales_csv(root)
            config=Config(sales_profile_path=str(path),sales_quarter='20251').validate()
            profiles=purpose_profiles(config,8)
            self.assertEqual(set(profiles),{'meal','dessert','mixed'})
            for values in profiles.values():
                self.assertEqual(len(values),8);self.assertAlmostEqual(sum(values),1.)
            # 11-14 holds most meal sales and 14-17 most cafe sales.
            self.assertGreater(profiles['meal'][0],profiles['meal'][4])
            self.assertGreater(profiles['dessert'][4],profiles['dessert'][0])
            # Long opening hours start earlier so the window stays inside one day.
            self.assertEqual(start_hour(config,16),8)
            self.assertEqual(len(purpose_profiles(config,16)['meal']),16)

    def test_missing_sales_file_and_invalid_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'sales profile'):Config(sales_profile_path='/nonexistent/seoul_sales.csv').validate()
        Config(sales_profile_path='/nonexistent/seoul_sales.csv').validate(require_source=False)
        with self.assertRaises(ValueError):Config(visitor_variation='normal').validate()
        for hour in (-1,24,11.5):
            with self.assertRaises(ValueError):Config(arrival_start_hour=hour).validate()

    def test_purpose_profiles_move_arrivals(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(),root);store=restore_store(run.state['cases'][0]['template'])
        people=customers(20,random.Random(1),store)
        def arrivals(profiles):
            result=simulate(people,store,Policy.initial('no_recommendation'),.5,SCENARIO,7,raw=True,decision_time_mode='equal',purpose_profiles=profiles)
            return [x['arrival'] for x in result['logs']]
        early={k:(1.,)+(0.,)*7 for k in ('meal','dessert','mixed')}
        late={k:(0.,)*7+(1.,) for k in ('meal','dessert','mixed')}
        self.assertTrue(all(0<=x<60 for x in arrivals(early)))
        self.assertTrue(all(420<=x<480 for x in arrivals(late)))
        self.assertEqual(arrivals(None),arrivals(None))

    def test_experiment_with_poisson_and_sales_profile_runs_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            config=small(algorithm='actor_critic',visitors=None,visitor_variation='poisson',
                         sales_profile_path=str(sales_csv(root)),models=['no_recommendation','fixed','contextual'],decision_time_mode='equal')
            full=Experiment(config,root,'full');full.run()
            paused=Experiment(config,root,'paused');paused.run(day_budget=1)
            paused=Experiment.resume(root,paused.id);paused.run()
            frame=deterministic(paused.records())
            pd.testing.assert_frame_equal(deterministic(full.records()),frame)
            self.assertEqual(set(frame['day']),{1,2,3})
            means={c['id']:round(float(c['features']['visitor_mean'])) for c in paused.state['cases']}
            daily=frame.drop_duplicates(['day','store_id'])
            self.assertTrue(any(r.visitors!=means[r.store_id] for r in daily.itertuples()))


if __name__=='__main__':unittest.main()
