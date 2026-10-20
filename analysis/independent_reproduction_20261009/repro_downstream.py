import numpy as np, pandas as pd, statlib
D=pd.read_csv('REPRO_DISEASE_PERIOD.csv'); F=pd.read_csv('REPRO_FACTOR_PERIOD_factor.csv')
DN={302:'腹泻病',322:'LRI',332:'脑膜炎'}; ON={1:'死亡',6:'发病'}
D['dz']=D.cause_id.map(DN)+D.measure_id.map(ON)
P=['1990–2000','2001–2010','2011–2023']
w=D.pivot_table(index=['location_id','location','dz'],columns='period',values=['category','residual','lower','upper','rate_ratio'],aggfunc='first')
prof=pd.DataFrame(index=w.index)
for i,p in enumerate(P,1):
    for v in ['category','residual','lower','upper','rate_ratio']: prof[f'p{i}_{v}']=w[(v,p)]
hi=(prof[[f'p{i}_category' for i in (1,2,3)]]=='稳健高尾').all(1)&(prof[[f'p{i}_lower' for i in (1,2,3)]]>0).all(1)
lo=(prof[[f'p{i}_category' for i in (1,2,3)]]=='稳健低尾').all(1)&(prof[[f'p{i}_upper' for i in (1,2,3)]]<0).all(1)
prof['group']=np.where(hi,'持续高尾',np.where(lo,'持续低尾','其他'))
prof=prof.reset_index(); prof.to_csv('REPRO_COUNTRY_DISEASE_PROFILES.csv',index=False,encoding='utf-8-sig')
# factor profiles
FW=F.pivot_table(index=['location_id','factor'],columns='period',values=['status','favourable_residual','eligible'],aggfunc='first')
fp=pd.DataFrame(index=FW.index)
for i,p in enumerate(P,1):
    fp[f'p{i}_status']=FW[('status',p)] if ('status',p) in FW else np.nan
    fp[f'p{i}_fav']=FW[('favourable_residual',p)]
    fp[f'p{i}_elig']=FW[('eligible',p)].fillna(False).astype(bool)
fp['n_eval']=fp[[f'p{i}_elig' for i in (1,2,3)]].sum(1)
fp['n_adv']=sum((fp[f'p{i}_status']=='明确不利') for i in (1,2,3))
fp['repeated_adverse']=(fp.n_eval>=2)&(fp.n_adv>=2)
fp=fp.reset_index(); fp.to_csv('REPRO_COUNTRY_FACTOR_PROFILES.csv',index=False,encoding='utf-8-sig')
CONTEXT=['income_ppp_pc','secondary_enrolment_pct','fertility_births_per_woman']
rows=[]
for dz in prof.dz.unique():
    g=prof[prof.dz==dz][['location_id','group']]
    for f in fp.factor.unique():
        x=fp[(fp.factor==f)&(fp.n_eval>=2)].merge(g,on='location_id')
        H=x[x.group=='持续高尾']; L=x[x.group=='持续低尾']; O=x[x.group=='其他']
        rec=dict(dz=dz,factor=f,high_n=len(H),high_adv=int(H.repeated_adverse.sum()),low_n=len(L),low_adv=int(L.repeated_adverse.sum()),other_n=len(O),other_adv=int(O.repeated_adverse.sum()),p=np.nan)
        if f not in CONTEXT and len(H)>=5 and len(L)>=5:
            a,b=rec['high_adv'],len(H)-rec['high_adv']; c,d_=rec['low_adv'],len(L)-rec['low_adv']
            rec['p']=statlib.fisher_exact(a,b,c,d_)
        rows.append(rec)
C=pd.DataFrame(rows); C['q']=statlib.bh(C.p.values); C.to_csv('REPRO_GROUP_CONTRASTS.csv',index=False,encoding='utf-8-sig')
# associations
arows=[]
Dp=D.pivot_table(index=['location_id','dz'],columns='period',values='residual').reset_index()
Fp=F[F.eligible].pivot_table(index=['location_id','factor'],columns='period',values='favourable_residual').reset_index()
for dz in Dp.dz.unique():
    dd=Dp[Dp.dz==dz]
    for f in Fp.factor.unique():
        ff=Fp[Fp.factor==f]; m=ff.merge(dd,on='location_id',suffixes=('_f','_d'))
        wins={'early':(m['1990–2000_f'],m['1990–2000_d']),'middle':(m['2001–2010_f'],m['2001–2010_d']),'late':(m['2011–2023_f'],m['2011–2023_d']),
              'early_late':(m['2011–2023_f']-m['1990–2000_f'],m['2011–2023_d']-m['1990–2000_d']),'middle_late':(m['2011–2023_f']-m['2001–2010_f'],m['2011–2023_d']-m['2001–2010_d'])}
        for wn,(x,y) in wins.items():
            rho,p,n=statlib.spearman(x,y)
            ok=n>=10 and np.nanstd(x)>0
            arows.append(dict(dz=dz,factor=f,window=wn,n=n,rho=rho if ok else np.nan,p=p if (ok and n>=20 and f not in CONTEXT) else np.nan))
A=pd.DataFrame(arows); A['q']=statlib.bh(A.p.values); A.to_csv('REPRO_ASSOCIATIONS.csv',index=False,encoding='utf-8-sig')
print('done', C.p.notna().sum(), A.p.notna().sum())
