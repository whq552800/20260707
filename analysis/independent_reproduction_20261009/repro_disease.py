"""Independent reproduction of disease reference models, period residuals,
conditional intervals and interval-then-quartile classification, from the raw annual panel."""
import numpy as np, pandas as pd
from paths import PANEL
import statlib
P=PANEL
d=pd.read_csv(P)
d=d[d.rate_valid & (d.gbd_rate_per_100k>0)].copy()
d['y']=np.log(d.gbd_rate_per_100k)
PER=[(1990,2000,'1990–2000'),(2001,2010,'2001–2010'),(2011,2023,'2011–2023')]
def per(y):
    for a,b,n in PER:
        if a<=y<=b: return n
d['period']=d.year.map(per)

def ns_basis(x, knots, bk=(0.0,1.0)):
    # natural cubic spline, truncated-power form (ESL 5.4), df = len(knots)+1
    k=np.r_[bk[0],knots,bk[1]]; K=len(k)
    def dk(j):
        return (np.clip(x-k[j],0,None)**3-np.clip(x-k[-1],0,None)**3)/(k[-1]-k[j])
    cols=[x]+[dk(j)-dk(K-2) for j in range(K-2)]
    return np.column_stack(cols)

out=[]; fitinfo=[]; annual=[]
sdi_all=d.drop_duplicates(['location_id','year']).sdi
knots=np.quantile(sdi_all,[0.25,0.5,0.75])
years=sorted(d.year.unique())
for (c,m),g in d.groupby(['cause_id','measure_id']):
    g=g.sort_values(['location_id','year']).reset_index(drop=True)
    B=ns_basis(g.sdi.values,knots)
    tc=((g.year-2006.5)/33).values[:,None]
    YF=pd.get_dummies(g.year,prefix='y',drop_first=True).astype(float).values
    X=np.column_stack([np.ones(len(g)),YF,B,B*tc])
    y=g.y.values
    XtX_inv=np.linalg.pinv(X.T@X); b=XtX_inv@X.T@y
    e=y-X@b; n,k=X.shape
    cl=g.location_id.values; G=len(np.unique(cl))
    meat=np.zeros((k,k))
    for loc,idx in g.groupby('location_id').indices.items():
        s=X[idx].T@e[idx]; meat+=np.outer(s,s)
    V=XtX_inv@meat@XtX_inv*(G/(G-1))*((n-1)/(n-k))
    g['fit']=X@b; g['res']=e
    fitinfo.append(dict(cause_id=c,measure_id=m,n=n,k=k,r2=1-e.var()/y.var()))
    annual.append(g[['location_id','cause_id','measure_id','year']].assign(fitted_log_rate=g.fit.values,residual_log=g.res.values))
    tcrit=statlib.t_ppf(0.975,G-1)
    for (loc,p),idx in g.groupby(['location_id','period']).indices.items():
        a=X[idx].mean(0); se=np.sqrt(a@V@a)
        r=g.res.values[idx].mean(); ob=g.y.values[idx].mean(); f=a@b
        out.append(dict(location_id=loc,location=g.location.values[idx[0]],cause_id=c,measure_id=m,period=p,
            sdi=g.sdi.values[idx].mean(),n_years=len(idx),residual=r,lower=ob-(f+tcrit*se),upper=ob-(f-tcrit*se)))
r=pd.DataFrame(out)
r['direction']=np.where(r.lower>0,'pos',np.where(r.upper<0,'neg','undetermined'))
def classify(r, keys):
    """Interval first, then quartiles of the resolved residuals within each disease, outcome and period."""
    res=r.residual.where(r.direction!='undetermined'); by=[r[k] for k in keys]
    q1=res.groupby(by).transform(lambda x: np.nanquantile(x,0.25)); q3=res.groupby(by).transform(lambda x: np.nanquantile(x,0.75))
    cat=np.where(r.direction=='undetermined','参照不确定',np.where(r.residual<q1,'稳健低尾',np.where(r.residual>q3,'稳健高尾','明确偏离但居中')))
    return r.assign(category=cat,q1=q1,q3=q3)
r=classify(r,['cause_id','measure_id','period'])
r['rate_ratio']=np.exp(r.residual)
r.to_csv('REPRO_DISEASE_PERIOD.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(fitinfo).to_csv('REPRO_MODEL_FIT.csv',index=False)
pd.concat(annual).to_csv('REPRO_ANNUAL_FIT.csv',index=False)   # annual fitted log rates and residuals
print(pd.DataFrame(fitinfo).to_string()); print('knots',knots)
