"""Run the whole synthetic demo with one command; never starts an FE solver."""
import argparse
import csv
import json
from pathlib import Path
from . import data,model,pipeline
from .predict import predict_csv


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(output,seed=20261011,plot=True):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    sample=data.generate(seed=seed); split=data.split_groups(sample['groups'],seed=seed)
    fitted,cv=pipeline.fit(sample,split,seed=seed)
    metrics,test,prediction,summary,group_prediction=pipeline.evaluate(fitted,sample)
    model.save(fitted,output/'model.json')
    rows=[]
    membership={g:k for k,groups in split.items() for g in groups}
    for i,g in enumerate(sample['groups']):
        rows.append({'data_kind':'SYNTHETIC_DEMONSTRATION','scenario_id':int(g),'split':membership[int(g)],
            **dict(zip(data.FEATURES,map(float,sample['x'][i]))),**dict(zip(data.RESPONSES,map(float,sample['y'][i])))})
    write_csv(output/'synthetic_dataset.csv',rows)
    write_csv(output/'cv_results.csv',cv); write_csv(output/'test_metrics.csv',metrics)
    cases=[]
    for i,g in enumerate(test['groups']):
        row={'data_kind':'SYNTHETIC_DEMONSTRATION','scenario_id':int(g)}
        for r,name in enumerate(data.RESPONSES):
            row[f'{name}_observed']=float(test['y'][i,r])
            for key in ('mean','lower_90','upper_90','lower_95','upper_95'):
                row[f'{name}_{key}']=float(prediction[key][i,r])
        cases.append(row)
    write_csv(output/'test_predictions.csv',cases)
    inputs=[dict(zip(data.FEATURES,v)) for v in [[1.,1.,1.],[.9,1.2,1.1],[1.1,.8,.9]]]
    write_csv(output/'prediction_inputs.csv',inputs)
    predict_csv(output/'model.json',output/'prediction_inputs.csv',output/'example_predictions.csv')
    report={'data_kind':'SYNTHETIC_DEMONSTRATION','seed':seed,'scenario_count':100,'repeats_per_scenario':30,
        'case_count':len(sample['x']),'partitions':{k:len(v) for k,v in split.items()},
        'responses':metrics,'data_source':'Generated algebraic relationships plus Gaussian noise; no actual FE or automotive data',
        'model_selection':'Five-fold training-scenario CV; no test tuning','figures_generated':bool(plot)}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if plot:
        from .plots import create_figures
        create_figures(output/'figures',metrics,summary,group_prediction,test,prediction,cv)
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts'))
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--no-plots',action='store_true',help='NumPy-only run, no matplotlib needed')
    args=parser.parse_args()
    run(args.output,seed=args.seed,plot=not args.no_plots)


if __name__=='__main__': main()
