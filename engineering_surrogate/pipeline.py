"""Fit on training scenarios, calibrate on calibration scenarios, evaluate once."""
import numpy as np
from . import data,model


def fit(sample,split,seed=20261011):
    data.validate_split(sample['groups'],split)
    # Only training observations are summarised for fitting and CV. Test labels
    # are intentionally not read by this function, even for scale estimation.
    mask=np.isin(sample['groups'],split['train'])
    training={k:v[mask] for k,v in sample.items()}
    summary=data.summarize(training)
    x,mean,scale=summary['x'],summary['mean'],summary['scale']
    fold_ids=np.array_split(np.random.default_rng(seed).permutation(len(x)),5)
    if min(map(len,fold_ids))<2: raise ValueError('Need enough training scenarios for five folds')
    records=[]; fitted_models=[]
    cmask=np.isin(sample['groups'],split['calibration'])
    cx,cy,cgroups=sample['x'][cmask],sample['y'][cmask],sample['groups'][cmask]
    # Validate calibration structure separately; no frozen-test values inspected.
    data.summarize({k:v[cmask] for k,v in sample.items()})
    for response,name in enumerate(data.RESPONSES):
        candidates=[]
        for degree in (1,2,3):
            for alpha in (.01,.1,1.):
                oof=np.empty(len(x))
                for valid in fold_ids:
                    train=np.setdiff1d(np.arange(len(x)),valid)
                    fitted=model.fit_ridge(x[train],mean[train,response],degree=degree,alpha=alpha)
                    oof[valid]=model.ridge_predict(fitted,x[valid])
                rmse=float(np.sqrt(np.mean((mean[:,response]-oof)**2)))
                row={'response':name,'degree':degree,'alpha':alpha,'cv_scenario_mean_rmse':rmse}
                records.append(row); candidates.append(row)
        best=min(candidates,key=lambda r:r['cv_scenario_mean_rmse'])
        point=model.fit_ridge(x,mean[:,response],degree=best['degree'],alpha=best['alpha'])
        # Scale model is intentionally fixed and simple, not selected on test data.
        log_scale=model.fit_ridge(x,np.log(scale[:,response]),degree=1,alpha=.1)
        centre=model.ridge_predict(point,cx); sd=np.exp(model.ridge_predict(log_scale,cx))
        scores=np.abs(cy[:,response]-centre)/sd
        fitted_models.append({'response':name,'mean':point,'log_scale':log_scale,
            'q90':model.hcp_quantile(scores,cgroups,.1),'q95':model.hcp_quantile(scores,cgroups,.05),
            'cv_scenario_mean_rmse':best['cv_scenario_mean_rmse']})
    fitted={'schema_version':1,'data_kind':'SYNTHETIC_DEMONSTRATION','seed':int(seed),
            'features':data.FEATURES,'responses':data.RESPONSES,'bounds':data.BOUNDS,'models':fitted_models,
            'split':split,'selection':'Five-fold CV on training scenario means only'}
    return fitted,records


def evaluate(fitted,sample):
    mask=np.isin(sample['groups'],fitted['split']['test'])
    test={k:v[mask] for k,v in sample.items()}
    summary=data.summarize(test)
    prediction=model.predict(fitted,test['x'])
    group_prediction=model.predict(fitted,summary['x'])
    metrics=[]
    for r,name in enumerate(data.RESPONSES):
        observed=summary['mean'][:,r]; predicted=group_prediction['mean'][:,r]
        residual=test['y'][:,r]-prediction['mean'][:,r]
        denominator=float(np.sum((observed-observed.mean())**2))
        row={'response':name,'scenario_mean_r2':float(1-np.sum((observed-predicted)**2)/denominator),
            'scenario_mean_rmse':float(np.sqrt(np.mean((observed-predicted)**2))),
            'realisation_rmse':float(np.sqrt(np.mean(residual**2))),'data_kind':'SYNTHETIC_DEMONSTRATION'}
        for level in (90,95):
            lower=prediction[f'lower_{level}'][:,r]; upper=prediction[f'upper_{level}'][:,r]
            covered=(test['y'][:,r]>=lower)&(test['y'][:,r]<=upper)
            # Each scenario has equal total weight even if repeat counts differ.
            row[f'coverage_{level}']=float(np.mean([covered[test['groups']==g].mean() for g in summary['groups']]))
            row[f'mean_width_{level}']=float(np.mean([(upper-lower)[test['groups']==g].mean() for g in summary['groups']]))
        metrics.append(row)
    return metrics,test,prediction,summary,group_prediction
