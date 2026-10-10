import csv
import tempfile
import unittest
from pathlib import Path
from engineering_surrogate import run,predict


class InterfaceTests(unittest.TestCase):
    def test_whole_demo_and_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=run.run(root,plot=False)
            self.assertEqual(report['case_count'],3000)
            self.assertEqual(predict.predict_csv(root/'model.json',root/'prediction_inputs.csv',root/'predicted.csv'),3)
            with (root/'predicted.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['data_kind'],'SYNTHETIC_DEMONSTRATION')
            for content in ['load_factor\n1\n','stiffness_factor,load_factor,geometry_factor\n',
                'stiffness_factor,load_factor,geometry_factor\n1,abc,1\n',
                'stiffness_factor,load_factor,geometry_factor\n1,1,1,extra\n',
                'stiffness_factor,load_factor,geometry_factor\n1,nan,1\n']:
                bad=root/'bad.csv'; bad.write_text(content,encoding='utf-8')
                with self.assertRaises(ValueError): predict.predict_csv(root/'model.json',bad,root/'bad_out.csv')
                self.assertFalse((root/'bad_out.csv').exists())
            with self.assertRaises(ValueError): predict.predict_csv(root/'model.json',root/'prediction_inputs.csv',root/'prediction_inputs.csv')


if __name__=='__main__': unittest.main()
