"""Small inspectable polynomial Ridge and calibrated prediction interface."""
import itertools
import json
from pathlib import Path
import numpy as np


def fit_ridge(x,y,degree=2,alpha=.1):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned training arrays required')
    if degree not in (1,2,3) or alpha<0: raise ValueError('Invalid degree or alpha')
    centre=x.mean(axis=0); scale=x.std(axis=0)
    if np.any(scale<=0): raise ValueError('Features must vary')
    terms=[list(t) for power in range(1,degree+1) for t in itertools.combinations_with_replacement(range(x.shape[1]),power)]
    z=(x-centre)/scale
    design=np.column_stack([np.ones(len(x))]+[z[:,t].prod(axis=1) for t in terms])
    penalty=np.eye(design.shape[1])*alpha; penalty[0,0]=0
    # Augmented least squares is more stable than inverting normal equations.
    coef=np.linalg.lstsq(np.vstack([design,np.sqrt(penalty)]),np.r_[y,np.zeros(len(penalty))],rcond=None)[0]
    return {'centre':centre.tolist(),'scale':scale.tolist(),'terms':terms,'coef':coef.tolist(),'degree':degree,'alpha':float(alpha)}


def ridge_predict(fitted,x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(fitted['centre']) or not np.isfinite(x).all():
        raise ValueError('Prediction features do not match')
    z=(x-np.asarray(fitted['centre']))/np.asarray(fitted['scale'])
    design=np.column_stack([np.ones(len(x))]+[z[:,t].prod(axis=1) for t in fitted['terms']])
    return design@np.asarray(fitted['coef'])


def hcp_quantile(scores,groups,alpha=.1):
    """Equal scenario mass, plus the finite-sample infinity atom for a new group."""
    scores=np.asarray(scores,dtype=float).reshape(-1); groups=np.asarray(groups).reshape(-1)
    if not len(scores) or groups.shape!=scores.shape or not np.isfinite(scores).all() or np.any(scores<0) or not 0<alpha<1:
        raise ValueError('Invalid calibration input')
    ids,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    n=len(ids)
    if n<2 or (1-alpha)>n/(n+1)+1e-12: raise ValueError('Too few groups for a finite interval')
    weights=1/((n+1)*counts[inverse])
    values=np.r_[scores,np.inf]; weights=np.r_[weights,1/(n+1)]
    order=np.argsort(values,kind='stable')
    index=np.searchsorted(np.cumsum(weights[order]),1-alpha-1e-12)
    result=float(values[order[min(index,len(order)-1)]])
    if not np.isfinite(result): raise ValueError('Calibration returned an infinite interval')
    return result


def predict(fitted,x):
    x=np.asarray(x,dtype=float); bounds=np.asarray(fitted['bounds'])
    if x.ndim!=2 or x.shape[1]!=len(bounds) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Need nonempty finite input with required feature columns')
    if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]): raise ValueError('Input outside demonstration domain')
    means=[]; scales=[]; q90=[]; q95=[]
    for entry in fitted['models']:
        means.append(ridge_predict(entry['mean'],x))
        scales.append(np.exp(ridge_predict(entry['log_scale'],x)))
        q90.append(entry['q90']); q95.append(entry['q95'])
    mean=np.column_stack(means); scale=np.column_stack(scales)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid model prediction')
    return {'mean':mean,'scale':scale,'lower_90':mean-scale*q90,'upper_90':mean+scale*q90,
            'lower_95':mean-scale*q95,'upper_95':mean+scale*q95}


def save(fitted,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(fitted,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load(path):
    fitted=json.loads(Path(path).read_text(encoding='utf-8'))
    if fitted.get('schema_version')!=1 or fitted.get('data_kind')!='SYNTHETIC_DEMONSTRATION':
        raise ValueError('Not a supported demonstration model')
    if len(fitted.get('models',[]))!=len(fitted.get('responses',[])) or not fitted['models']:
        raise ValueError('Malformed model response list')
    bounds=np.asarray(fitted.get('bounds',[]),dtype=float)
    if bounds.shape!=(len(fitted['features']),2) or not np.isfinite(bounds).all() or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('Malformed model bounds')
    for entry in fitted['models']:
        if not 0<=entry['q90']<=entry['q95']<np.inf: raise ValueError('Invalid interval quantiles')
        for key in ('mean','log_scale'):
            part=entry[key]; n=len(fitted['features'])
            if len(part['centre'])!=n or len(part['scale'])!=n or len(part['coef'])!=len(part['terms'])+1:
                raise ValueError('Malformed polynomial arrays')
            values=part['centre']+part['scale']+part['coef']
            if not np.isfinite(values).all() or min(part['scale'])<=0:
                raise ValueError('Invalid polynomial values')
            if any(not t or any(not isinstance(i,int) or i<0 or i>=n for i in t) for t in part['terms']):
                raise ValueError('Invalid polynomial terms')
    return fitted
