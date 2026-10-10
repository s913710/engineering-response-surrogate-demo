"""CSV deployment example: python -m engineering_surrogate.predict ..."""
import argparse
import csv
from pathlib import Path
import numpy as np
from . import model


def predict_csv(model_path,input_path,output_path):
    fitted=model.load(model_path)
    with Path(input_path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV headers')
        if not set(fitted['features'])<=set(reader.fieldnames): raise ValueError('Missing required feature columns')
        rows=list(reader)
        if not rows: raise ValueError('CSV contains no input rows')
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('CSV has malformed rows')
    try: x=np.asarray([[float(row[name]) for name in fitted['features']] for row in rows])
    except (TypeError,ValueError) as exc: raise ValueError('Features must be numeric') from exc
    prediction=model.predict(fitted,x)
    out=[]
    for i in range(len(rows)):
        row={'data_kind':'SYNTHETIC_DEMONSTRATION',**{name:rows[i][name] for name in fitted['features']}}
        for r,name in enumerate(fitted['responses']):
            for statistic in ('mean','lower_90','upper_90','lower_95','upper_95'):
                row[f'{name}_{statistic}']=float(prediction[statistic][i,r])
        out.append(row)
    output_path=Path(output_path)
    if output_path.resolve()==Path(input_path).resolve() or output_path.resolve()==Path(model_path).resolve():
        raise ValueError('Output must not overwrite input or model')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    return len(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try: count=predict_csv(args.model,args.input,args.output)
    except (ValueError,KeyError,OSError) as exc: parser.exit(2,f'Prediction error: {exc}\n')
    print(f'SYNTHETIC_DEMONSTRATION: predicted {count} rows -> {args.output}')


if __name__=='__main__': main()
