import copy
import json
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
import pandas as pd
from kiosk_research.engine import Experiment
from kiosk_research.model_board import ModelBoard,matched_points
from test_pareto_research import small


class ModelBoardTests(unittest.TestCase):
    def config(self):return small(algorithm='actor_critic',models=['no_recommendation','fixed','contextual'],decision_time_mode='equal',auto_cleanup=False)

    def test_control_pairing_rank_detail_and_readonly_server(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root);run.run()
            original=copy.deepcopy(run.state)
            board=ModelBoard(run,'2026-10-29T09:00:00Z',display=False,max_points=2)
            snap=board.snapshot()
            self.assertEqual(len(snap['cards']),16)
            gains=[c['rank']['reward'] for c in snap['cards']]
            self.assertEqual(gains,sorted(gains,reverse=True))
            key=snap['cards'][0]['id'];detail=board.detail(key,'0')
            self.assertEqual(detail['lambda'],'0.0')
            self.assertEqual(len(detail['series']),2)
            self.assertEqual(detail['rank']['days'],[3])
            for p in detail['series']:
                self.assertAlmostEqual(sum(p['weights']),1.)
                self.assertIn('baseline_customer',p)
            port=board.start_server()
            try:
                with urlopen(f'http://127.0.0.1:{port}/api/index') as response:
                    self.assertEqual(json.load(response)['experiment_id'],run.id)
                from urllib.parse import quote
                with urlopen(f'http://127.0.0.1:{port}/api/model?id={quote(key)}&lambda=0') as response:
                    self.assertEqual(json.load(response)['lambda'],'0.0')
                with self.assertRaises(HTTPError):urlopen(f'http://127.0.0.1:{port}/../../metadata.json')
                with urlopen(f'http://127.0.0.1:{port}/') as response:
                    page=response.read().decode()
                    self.assertIn('모든 모델로 돌아가기',page)
                    self.assertIn('완료된 동결 평가 주기',page)
            finally:board.server.shutdown();board.server.server_close()
            self.assertEqual(run.state,original)

    def test_partial_points_never_enter_frozen_rank(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root);run.run(day_budget=1)
            board=ModelBoard(run,None,display=False)
            self.assertTrue(all(c['rank'] is None for c in board.snapshot()['cards']))
            def progress(day,done,total):
                board.progress(day,done,total)
                self.assertTrue(all(c['rank'] is None for c in board.snapshot()['cards']))
                self.assertTrue(any(c['provisional'] for c in board.snapshot()['cards']))
            run.run(day_budget=1,progress=progress,day_callback=board.completed)
            self.assertTrue(all(c['rank'] is None for c in board.snapshot()['cards']))
            run.run(day_callback=board.completed)
            self.assertTrue(all(c['rank'] is not None for c in board.snapshot()['cards']))
            before=board.snapshot()['updated_at_utc']
            self.assertEqual(before,board.snapshot()['updated_at_utc'])

    def test_missing_matched_control_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run=Experiment(self.config(),root);run.run(day_budget=1)
            frame=run.export_metrics()
            with self.assertRaises(ValueError):matched_points(frame[frame.model.ne('no_recommendation')])


if __name__=='__main__':unittest.main()
