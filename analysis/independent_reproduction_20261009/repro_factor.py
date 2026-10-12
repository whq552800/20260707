"""Independent reproduction of the 19 background-factor reference models and period residuals."""
import numpy as np, pandas as pd, statlib, sys
from paths import PANEL
from repro_common import ns_basis, period_of
KNOTMODE=sys.argv[1] if len(sys.argv)>1 else 'factor'
d=pd.read_csv(PANEL)
ly=d.drop_duplicates(['location_id','year']).copy()
LOGIT=['basic_water_pct','basic_sanitation_pct','clean_cooking_pct','dtp3_pct','hib3_pct','pcv3_pct','stunting_u5_pct','low_birthweight_pct','undernourishment_pct','uhc_service_index','secondhand_smoke_sev']
LOG=['fertility_births_per_woman','health_expenditure_pc_current_usd','income_ppp_pc','wdi_pm25_annual_mean_ug_m3']
LOG1P=['hospital_beds_per1000','physicians_per1000','nurses_midwives_per1000','secondary_enrolment_pct']
LOWER_BETTER=['stunting_u5_pct','low_birthweight_pct','undernourishment_pct','wdi_pm25_annual_mean_ug_m3','secondhand_smoke_sev','fertility_births_per_woman']
CONTEXT=['income_ppp_pc','secondary_enrolment_pct','fertility_births_per_woman']
def fwd(v,f):
    if f in LOGIT: p=np.clip(v,0.1,99.9)/100; return np.log(p/(1-p))
    if f in LOG: return np.log(v)
    return np.log1p(v)
def inv(z,f):
    if f in LOGIT: return 100/(1+np.exp(-z))
    if f in LOG: return np.exp(z)
    return np.expm1(z)
def dinv(z,f):
    if f in LOGIT: s=1/(1+np.exp(-z)); return 100*s*(1-s)
    return np.exp(z)
knots_all=np.quantile(ly.sdi,[0.25,.5,.75])
rows=[]
for f in LOGIT+LOG+LOG1P:
    g=ly[['location_id','location','year','sdi',f]].dropna().sort_values(['location_id','year']).reset_index(drop=True)
    if f in LOG: g=g[g[f]>0]
    g=g.reset_index(drop=True)
    kn=np.quantile(g.sdi,[0.25,.5,.75]) if KNOTMODE=='factor' else knots_all
    B=ns_basis(g.sdi.values,kn); tc=((g.year-2006.5)/33).values[:,None]
    YF=pd.get_dummies(g.year,prefix='y',drop_first=True).astype(float).values
    X=np.column_stack([np.ones(len(g)),YF,B,B*tc]); z=fwd(g[f].values,f)
    XtX=np.linalg.pinv(X.T@X); b=XtX@X.T@z; e=z-X@b; n,k=X.shape
    G=g.location_id.nunique(); meat=np.zeros((k,k))
    for _,idx in g.groupby('location_id').indices.items():
        s=X[idx].T@e[idx]; meat+=np.outer(s,s)
    V=XtX@meat@XtX*(G/(G-1))*((n-1)/(n-k)); tq=statlib.t_ppf(0.975,G-1)
    fz=X@b; fit=inv(fz,f); g['res']=g[f].values-fit; g['period']=g.year.map(period_of)
    for (loc,p),idx in g.groupby(['location_id','period']).indices.items():
        grad=(dinv(fz[idx],f)[:,None]*X[idx]).mean(0); se=np.sqrt(grad@V@grad)
        r=g.res.values[idx].mean(); sign=-1 if f in LOWER_BETTER else 1
        lo,hi=r-tq*se,r+tq*se
        flo,fhi=(lo,hi) if sign==1 else (-hi,-lo)
        rows.append(dict(location_id=loc,location=g.location.values[idx[0]],factor=f,period=p,n_years=len(idx),
            raw_residual=r,raw_lower=lo,raw_upper=hi,favourable_residual=sign*r,favourable_lower=flo,favourable_upper=fhi))
r=pd.DataFrame(rows)
r['eligible']=r.n_years>=2
r['status']=np.where(~r.eligible,'不可评价',np.where(r.favourable_lower>0,'明确有利',np.where(r.favourable_upper<0,'明确不利','方向不确定')))
r.to_csv(f'REPRO_FACTOR_PERIOD_{KNOTMODE}.csv',index=False,encoding='utf-8-sig')
