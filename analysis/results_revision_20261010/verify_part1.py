"""核验结果整合稿第一部分的数字（以当前 REPRO_DISEASE_PERIOD 为准）。"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,'../independent_reproduction_20261009')
from statlib import spearman
D=pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
dzn={302:'腹泻',322:'LRI',332:'脑膜炎'}; mn={1:'死亡',6:'发病'}
D['dz']=D.cause_id.map(dzn); D['m']=D.measure_id.map(mn)
P=['1990–2000','2001–2010','2011–2023']
out=[]
def W(k,v): out.append((k,v)); print(k,v)
# 1 跨病种相关
for m in ['死亡','发病']:
  for p in P:
    w=D[(D.m==m)&(D.period==p)].pivot(index='location_id',columns='dz',values='residual')
    rs=[spearman(w[a].values,w[b].values)[0] if isinstance(spearman(w[a].values,w[b].values),tuple) else spearman(w[a].values,w[b].values) for a,b in [('腹泻','LRI'),('腹泻','脑膜炎'),('LRI','脑膜炎')]]
    W(f'跨病种{m}相关 {p} 腹泻LRI/腹泻脑膜炎/LRI脑膜炎',[round(float(r),3) for r in rs])
# 2 晚期方向全明确地点
L=D[D.period==P[2]].copy()
L['s']=L.direction.map({'pos':1,'neg':-1}).fillna(0)
w=L.pivot_table(index='location_id',columns=['m','dz'],values='s')
ok=(w!=0).all(axis=1); ws=w[ok]
W('晚期六结局方向全明确地点数',int(ok.sum()))
for m in ['死亡','发病']:
  c=(ws[m].nunique(axis=1)==1).mean()*100; W(f'{m}三病方向一致%',round(c,1))
# 3 发病-死亡
for dz in ['腹泻','LRI','脑膜炎']:
  rr=[]
  for p in P:
    a=D[(D.dz==dz)&(D.period==p)].pivot(index='location_id',columns='m',values='residual')
    r=spearman(a['发病'].values,a['死亡'].values); rr.append(round(float(r[0] if isinstance(r,tuple) else r),3))
  b=L[L.dz==dz].pivot(index='location_id',columns='m',values='s'); b=b[(b!=0).all(axis=1)]
  opp=(b['发病']!=b['死亡']); io=((b['发病']==1)&(b['死亡']==-1)).sum(); oi=((b['发病']==-1)&(b['死亡']==1)).sum()
  c=L[L.dz==dz].pivot(index='location_id',columns='m',values='category')
  strict=(((c['发病']=='稳健高尾')&(c['死亡']=='稳健低尾'))|((c['发病']=='稳健低尾')&(c['死亡']=='稳健高尾'))).sum()
  W(f'{dz} 发病-死亡相关三期',rr); W(f'{dz} 晚期方向相反',f'{opp.sum()}/{len(b)} ({opp.mean()*100:.1f}%) 发高死低{io} 发低死高{oi} 严格尾部相反{strict}')
# 4 Australia / Greenland
pr=pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
for loc in ['Australia','Greenland']:
  x=pr[(pr.location==loc)&(pr.dz.str.startswith('腹泻'))]
  print(x[['dz','p1_category','p2_category','p3_category','p3_rate_ratio']].to_string())
# 5 SDI 上升
s=D.drop_duplicates(['location_id','period']).pivot(index='location_id',columns='period',values='sdi')
W('SDI逐期上升地点', f'{int(((s[P[1]]>s[P[0]])&(s[P[2]]>s[P[1]])).sum())}/{len(s)}')
for dz in ['腹泻','LRI','脑膜炎']:
  for m in ['发病','死亡']:
    a=D[(D.dz==dz)&(D.m==m)].pivot(index='location_id',columns='period',values='residual')
    r=spearman(a[P[0]].values,a[P[2]].values); W(f'{dz}{m} 早晚期排名相关',round(float(r[0] if isinstance(r,tuple) else r),3))
# 6 尾部保留
e=D[D.period==P[0]].set_index(['location_id','cause_id','measure_id']).category
l=D[D.period==P[2]].set_index(['location_id','cause_id','measure_id']).category
j=pd.concat([e,l],axis=1,keys=['e','l'])
for t,o in [('稳健高尾','稳健低尾'),('稳健低尾','稳健高尾')]:
  x=j[j.e==t]; W(f'早期{t}',f'{len(x)} 晚期仍{(x.l==t).sum()} ({(x.l==t).mean()*100:.1f}%) 转{o} {(x.l==o).sum()}')
# 7 持续高尾
pp=pr.copy(); pp['ph']=(pp[['p1_category','p2_category','p3_category']]=='稳健高尾').all(axis=1)
pp['pl']=(pp[['p1_category','p2_category','p3_category']]=='稳健低尾').all(axis=1)
g=pp[pp.ph].groupby('location').size()
W('持续高尾记录/地点',f'{pp.ph.sum()} / {len(g)}; 1-2结局 {int((g<=2).sum())} ({(g<=2).mean()*100:.1f}%); 6结局 {list(g[g==6].index)}')
g2=pp[pp.pl].groupby('location').size(); W('持续低尾记录/地点',f'{pp.pl.sum()} / {len(g2)}')
W('任一时期任一结局稳健高尾地点',D[D.category=='稳健高尾'].location_id.nunique())
pd.DataFrame(out,columns=['item','value']).to_csv('VERIFY_PART1.csv',index=False,encoding='utf-8-sig')
