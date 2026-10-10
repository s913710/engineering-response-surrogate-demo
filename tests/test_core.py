"""Behaviour checks for the public synthetic demonstration."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from engineering_surrogate import data, model, pipeline


class DataTests(unittest.TestCase):
    def test_reproducibility_and_disjoint_complete_split(self):
        first = data.generate(seed=42)
        second = data.generate(seed=42)
        np.testing.assert_array_equal(first['y'], second['y'])
        split = data.split_groups(first['groups'], seed=42)
        self.assertEqual([len(split[k]) for k in ('train','calibration','test')], [60,20,20])
        sets = [set(v) for v in split.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(first['groups']))
        bad = {k:list(v) for k,v in split.items()}
        bad['test'][0] = bad['train'][0]
        with self.assertRaises(ValueError): data.validate_split(first['groups'],bad)

    def test_group_features_cannot_change_between_realisations(self):
        sample = data.generate(seed=7)
        sample['x'][1,0] += .01
        with self.assertRaises(ValueError): data.summarize(sample)


class ModelTests(unittest.TestCase):
    def test_polynomial_recovers_known_quadratic(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(-1,1,(90,3))
        y = 3+2*x[:,0]**2-.7*x[:,0]*x[:,1]+x[:,2]
        fitted = model.fit_ridge(x,y,degree=2,alpha=0)
        np.testing.assert_allclose(model.ridge_predict(fitted,x),y,atol=1e-9)

    def test_hierarchical_weights_do_not_favour_larger_groups(self):
        groups=np.repeat(np.arange(20),2)
        scores=np.repeat(np.arange(20,dtype=float),2)
        before=model.hcp_quantile(scores,groups,alpha=.1)
        extra=np.repeat(0.,400)
        after=model.hcp_quantile(np.r_[scores,extra],np.r_[groups,np.zeros(400)],alpha=.1)
        self.assertEqual(before,after)
        self.assertEqual(before,18.)
        with self.assertRaises(ValueError): model.hcp_quantile(np.arange(8.),np.arange(8),alpha=.1)

    def test_model_domain_finite_values_and_serialisation(self):
        sample=data.generate(seed=19)
        split=data.split_groups(sample['groups'],seed=19)
        fitted,_=pipeline.fit(sample,split)
        x=sample['x'][:4]
        expected=model.predict(fitted,x)
        self.assertTrue(np.all(expected['lower_95']<=expected['lower_90']))
        self.assertTrue(np.all(expected['upper_95']>=expected['upper_90']))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json'
            model.save(fitted,path)
            loaded=model.load(path)
            np.testing.assert_array_equal(expected['mean'],model.predict(loaded,x)['mean'])
        for bad in [np.array([[2.,1.,1.]]),np.array([[np.nan,1.,1.]]),np.zeros((2,2)),np.empty((0,3))]:
            with self.assertRaises(ValueError): model.predict(fitted,bad)

    def test_test_labels_cannot_change_fit_or_calibration(self):
        sample=data.generate(seed=21)
        split=data.split_groups(sample['groups'],seed=21)
        first,_=pipeline.fit(sample,split)
        changed={k:v.copy() for k,v in sample.items()}
        test=np.isin(changed['groups'],split['test'])
        changed['y'][test] += 1000
        second,_=pipeline.fit(changed,split)
        self.assertEqual(first,second)


if __name__=='__main__': unittest.main()
