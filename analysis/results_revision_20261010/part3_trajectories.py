"""第三部分：中期→晚期服务缺口变化与疾病相对位置变化（重算R3并给出比例）；晚期高SDI的LRI发病稳健高尾名单。"""
import pandas as pd, numpy as np
import sys; sys.path.insert(0,'../independent_reproduction_20261009')
from statlib import rankdata
def wilson(k,n,z=1.959964):
    if n==0: return np.nan,np.nan
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def newcombe(k1,n1,k2,n2):
    """两独立比例之差 p1−p2 的 Newcombe 混合得分区间（百分点）。"""
    if n1==0 or n2==0: return np.nan,np.nan
    p1,p2=k1/n1,k2/n2; l1,u1=wilson(k1,n1); l2,u2=wilson(k2,n2); d=p1-p2
    return round(100*(d-np.sqrt((p1-l1)**2+(u2-p2)**2)),1),round(100*(d+np.sqrt((u1-p1)**2+(p2-l2)**2)),1)
def boot_rho(x,y,B=2000,seed=20261011):
    r=lambda i: np.corrcoef(rankdata(x[i]),rankdata(y[i]))[0,1]; idx=np.arange(len(x)); est=r(idx)
    if B==0: return est,np.nan,np.nan
    rng=np.random.default_rng(seed); bs=[r(rng.integers(0,len(x),len(x))) for _ in range(B)]
    return est,round(float(np.quantile(bs,.025)),6),round(float(np.quantile(bs,.975)),6)
D=pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
F=pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
T=pd.read_csv('../SDI_PERIOD_TIERS.csv')
M,L='2001–2010','2011–2023'
CAUSE={'腹泻病':302,'LRI':322,'脑膜炎':332}
from paths import PANEL
_p=pd.read_csv(PANEL,usecols=['location_id','cause_id','measure_id','year','gbd_rate_per_100k']); _p['period']=np.where(_p.year<=2000,'1990–2000',np.where(_p.year<=2010,M,L))
RATE=_p.groupby(['cause_id','measure_id','location_id','period']).gbd_rate_per_100k.mean().unstack('period')   # 各期平均GBD率（绝对水平）
out=[]
for f,dz,mid in [('basic_water_pct','腹泻病',1),('basic_sanitation_pct','腹泻病',1),('uhc_service_index','腹泻病',1),('dtp3_pct','LRI',1),('uhc_service_index','LRI',1),('stunting_u5_pct','LRI',1)]:
  for meas,mn in [(1,'死亡'),(6,'发病')]:
    fw=F[F.factor==f].pivot(index='location_id',columns='period',values=['favourable_residual','status'])
    fw=fw[(fw[('status',M)]=='明确不利')&fw[('status',L)].isin(['明确不利','明确有利','方向不确定','已达普及'])]
    nar=fw[('favourable_residual',L)]>fw[('favourable_residual',M)]
    rw=D[(D.cause_id==CAUSE[dz])&(D.measure_id==meas)].pivot(index='location_id',columns='period',values='residual')
    imp=(rw[L]<rw[M]).reindex(fw.index)
    a,b=imp[nar],imp[~nar]
    # 2026-10-11（审稿意见）：比例差的 Newcombe 区间；变化量的连续关联（Spearman，按地点重抽样区间）；起点缺口的检查（均值回归）
    d_f=(fw[('favourable_residual',L)]-fw[('favourable_residual',M)]).astype(float); d_r=(rw[L]-rw[M]).reindex(fw.index).astype(float)
    base=fw[('favourable_residual',M)].astype(float); ok=d_r.notna()
    lo,hi=newcombe(int(a.sum()),len(a),int(b.sum()),len(b))
    rho,rlo,rhi=boot_rho(d_f[ok].values,d_r[ok].values)
    rb,_,_=boot_rho(base[ok].values,d_r[ok].values,B=0)
    out.append(dict(factor=f,outcome=dz+mn,n=len(fw),narrowed=int(nar.sum()),narrowed_improved=int(a.sum()),pct_improved_if_narrowed=round(100*a.mean(),1),
        not_narrowed=int((~nar).sum()),not_narrowed_improved=int(b.sum()),pct_improved_if_not=round(100*b.mean(),1) if len(b) else np.nan,
        diff_pp=round(100*(a.mean()-b.mean()),1) if len(b) else np.nan,diff_lo=lo,diff_hi=hi,rho_change=round(rho,6),rho_lo=rlo,rho_hi=rhi,
        median_baseline_gap_narrowed=round(base[nar].median(),2),median_baseline_gap_not=round(base[~nar].median(),2) if len(b) else np.nan,
        rho_baseline_vs_disease_change=round(rb,6),median_disease_change=round(d_r.median(),3),
        pct_absolute_rate_fell=round(100*(RATE.loc[(CAUSE[dz],meas)][L]<RATE.loc[(CAUSE[dz],meas)][M]).reindex(fw.index).mean(),1),
        median_rate_change_pct=round(100*((RATE.loc[(CAUSE[dz],meas)][L]/RATE.loc[(CAUSE[dz],meas)][M]).reindex(fw.index)-1).median(),1)))
O=pd.DataFrame(out); print(O.to_string()); O.to_csv('PART3_CATCHUP_VS_DISEASE.csv',index=False,encoding='utf-8-sig')
tl=T[T.period==L].set_index('location_id').tier
x=D[(D.cause_id==322)&(D.measure_id==6)&(D.period==L)].copy(); x['tier']=x.location_id.map(tl)
h=x[(x.tier=='高SDI')]
print('高SDI晚期地点',len(h),'LRI发病稳健高尾',(h.category=='稳健高尾').sum())
print(h[h.category=='稳健高尾'][['location','rate_ratio']].sort_values('rate_ratio',ascending=False).round(2).to_string())
for loc in ['Singapore']:
  print(D[(D.location==loc)&(D.period==L)][['cause_id','measure_id','rate_ratio']].round(3).to_string())
