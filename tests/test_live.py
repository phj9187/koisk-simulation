from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.live import LiveDashboard,summarize
from kiosk_research.persistence import read_json
from test_pareto_research import small


class LiveTests(unittest.TestCase):
    def test_expired_deadline_starts_no_day(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(),root)
            run.run(deadline_utc='2000-01-01T00:00:00Z')
            self.assertEqual(run.state['day'],0)
            self.assertEqual(run.state['stop_reason'],'deadline_reached')
            with self.assertRaises(ValueError):run.run(deadline_utc='2026-10-28T12:00:00')

    def test_deadline_preserves_partial_store_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(visitors=0),root)
            with patch('kiosk_research.engine.datetime',wraps=datetime) as clock:
                clock.now.return_value=datetime(2026,10,28,2,tzinfo=timezone.utc)
                def progress(day,done,total):
                    clock.now.return_value=datetime(2026,10,28,4,tzinfo=timezone.utc)
                run.run(deadline_utc='2026-10-28T03:00:00Z',progress=progress)
            self.assertEqual(run.state['day'],0)
            self.assertEqual(len(list((run.checkpoints/'partial_000001').glob('store_*.json'))),1)
            resumed=Experiment.resume(root,run.id)
            with patch('kiosk_research.engine.store_day',wraps=__import__('kiosk_research.worker',fromlist=['store_day']).store_day) as worker:
                resumed.run(day_budget=1,deadline_utc='2099-01-01T00:00:00Z')
            self.assertEqual(worker.call_count,7)
            self.assertEqual(resumed.state['day'],1)

    def test_live_partial_final_and_resume_are_readonly(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(small(),root)
            dashboard=LiveDashboard(run,'2026-10-28T03:00:00Z',display=False)
            seen=[]
            def progress(day,done,total):
                dashboard.progress(day,done,total)
                status=read_json(run.directory/'live/status.json')
                self.assertTrue(status['provisional_last_point'])
                self.assertEqual(status['completed_stores'],done)
                seen.append(done)
            run.run(day_budget=1,progress=progress,day_callback=dashboard.completed)
            self.assertEqual(seen,list(range(1,9)))
            frame=pd.read_csv(run.directory/'live/scores.csv')
            pd.testing.assert_frame_equal(frame,summarize(run.export_metrics()),check_dtype=False)
            self.assertFalse(read_json(run.directory/'live/status.json')['provisional_last_point'])
            page=(run.directory/'live/index.html').read_text(encoding='utf-8')
            self.assertIn('<svg',page);self.assertIn('Customer score',page)
            restored=LiveDashboard(Experiment.resume(root,run.id),'2026-10-28T03:00:00Z',display=False)
            pd.testing.assert_frame_equal(restored.history,summarize(run.export_metrics()))


if __name__=='__main__':unittest.main()
