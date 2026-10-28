import unittest
import numpy as np
import torch
from ncsp_ip import decoder
from ncsp_ip.branches import Branch
from ncsp_ip.prior import normalized, project


class CoreTests(unittest.TestCase):
    def test_projection_preserves_order_and_ties(self):
        names = np.array([[.4, .4, .15, .05]], dtype=float)
        graph = np.array([[.1, .2, .25, .45]], dtype=float)
        fused, _ = project(names, (names + graph)/2)
        self.assertAlmostEqual(float(fused.sum()), 1.)
        self.assertEqual(fused[0, 0], fused[0, 1])
        self.assertGreater(fused[0, 1], fused[0, 2])
        self.assertGreater(fused[0, 2], fused[0, 3])

    def test_lambda_zero_checkpoint_round_trip(self):
        decoder.DEV = 'cpu'
        torch.set_num_threads(2)
        A = np.array([[1,0,0], [0,1,0], [0,0,0]], dtype=np.float32)
        x = np.eye(3, dtype=np.float32)
        ann = np.zeros((3,92), dtype=np.float32)
        bundle = {}
        for arm in ['off','real']:
            torch.manual_seed(8)
            weights = Branch(3,3).state_dict()
            W = weights.pop('W')
            bundle['101/'+arm] = dict(state=weights, W=W)
        prior = np.full_like(A, 1/3)
        context = dict(A=A, x=x, ann=ann, G=normalized(A), prior=prior, bundle=bundle, arms=['off','real'])
        state, score = decoder.fit(dict(seed=7), 'full', 101, context, np.array([[0,1],[1,2]]))
        reconstructed = decoder.predict(context, {'101':state})
        np.testing.assert_allclose(reconstructed, score, atol=1e-7, rtol=1e-7)
        self.assertEqual(state['lambda_penalty'], 0.)
        self.assertTrue(all(row['penalty'] == 0 for row in state['trace']))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score > 0) & (score < 1)).all())
        self.assertLess(state['gradient_checks']['positive_logit_derivative'], 0)
        self.assertGreater(state['gradient_checks']['negative_logit_derivative'], 0)


if __name__ == '__main__':
    unittest.main()
