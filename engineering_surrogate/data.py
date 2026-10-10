"""Generate an explicitly synthetic grouped regression problem."""
import numpy as np

FEATURES = ['stiffness_factor', 'load_factor', 'geometry_factor']
RESPONSES = ['displacement_mm', 'stress_mpa']
BOUNDS = [[.8,1.2],[.5,1.5],[.8,1.2]]


def generate(seed=20261011, scenarios=100, repeats=30):
    """Illustrative algebraic relationships, not FE or measured responses."""
    if scenarios<10 or repeats<2: raise ValueError('Need at least 10 scenarios and 2 repeats')
    rng=np.random.default_rng(seed)
    bounds=np.asarray(BOUNDS)
    # A stratified design reduces gaps without importing external design tools.
    u=np.column_stack([(rng.permutation(scenarios)+rng.random(scenarios))/scenarios for _ in FEATURES])
    xs=bounds[:,0]+u*(bounds[:,1]-bounds[:,0])
    k,load,g=xs.T
    mean=np.column_stack([6*load*g**3/k,28*load*g**2/k**.2])
    scale=np.column_stack([.15+.18*(load-.5)+.1*abs(g-1),1+.9*load])
    x=np.repeat(xs,repeats,axis=0)
    y=np.repeat(mean,repeats,axis=0)+rng.normal(size=(scenarios*repeats,2))*np.repeat(scale,repeats,axis=0)
    return {'x':x,'y':y,'groups':np.repeat(np.arange(scenarios),repeats)}


def split_groups(groups,seed=20261011):
    ids=np.unique(groups)
    if len(ids)<10: raise ValueError('Need at least ten groups')
    ids=np.random.default_rng(seed).permutation(ids)
    n=len(ids); a=int(.6*n); b=int(.8*n)
    result={k:sorted(map(int,v)) for k,v in zip(('train','calibration','test'),(ids[:a],ids[a:b],ids[b:]))}
    validate_split(groups,result)
    return result


def validate_split(groups,split):
    if set(split)!= {'train','calibration','test'}: raise ValueError('Missing partitions')
    sets=[set(split[k]) for k in ('train','calibration','test')]
    if any(not s for s in sets) or any(len(split[k])!=len(set(split[k])) for k in split):
        raise ValueError('Partitions must be nonempty with unique IDs')
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise ValueError('Scenario leakage')
    if set.union(*sets)!=set(np.unique(groups)): raise ValueError('Split does not cover data')


def summarize(sample):
    x,y,groups=sample['x'],sample['y'],sample['groups']
    if x.ndim!=2 or x.shape[1]!=3 or y.shape!=(len(x),2) or groups.shape!=(len(x),):
        raise ValueError('Incorrect dataset shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite data')
    ids=np.unique(groups); features=[]; means=[]; scales=[]
    for group in ids:
        mask=groups==group
        if mask.sum()<2: raise ValueError('Each group needs repeated observations')
        if not np.allclose(x[mask],x[mask][0],rtol=0,atol=1e-12):
            raise ValueError('Features vary within a scenario')
        features.append(x[mask][0]); means.append(y[mask].mean(axis=0))
        scales.append(y[mask].std(axis=0,ddof=1))
    if np.any(np.asarray(scales)<=0): raise ValueError('Repeated responses need positive variation')
    return {'groups':ids,'x':np.asarray(features),'mean':np.asarray(means),'scale':np.asarray(scales)}
