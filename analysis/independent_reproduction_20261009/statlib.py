"""Small numpy-only statistics helpers (scipy unavailable in this environment)."""
import numpy as np, math
def t_pdf(x,df):
    return math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)*(1+x*x/df)**(-(df+1)/2)
def t_cdf(x,df):
    xs=np.linspace(0,abs(x),20001); v=np.array([t_pdf(t,df) for t in xs])
    area=np.trapezoid(v,xs) if hasattr(np,'trapezoid') else np.trapz(v,xs)
    return 0.5+np.sign(x)*area
def t_ppf(p,df):
    lo,hi=0.0,20.0
    for _ in range(80):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2
def rankdata(a):
    a=np.asarray(a,float); o=np.argsort(a,kind='mergesort'); r=np.empty(len(a)); 
    s=a[o]; i=0
    while i<len(a):
        j=i
        while j+1<len(a) and s[j+1]==s[i]: j+=1
        r[o[i:j+1]]=(i+j)/2+1; i=j+1
    return r
def spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); m=~(np.isnan(x)|np.isnan(y)); x,y=x[m],y[m]
    n=len(x)
    if n<3: return np.nan,np.nan,n
    rx,ry=rankdata(x),rankdata(y); rho=np.corrcoef(rx,ry)[0,1]
    if abs(rho)>=1: return rho,0.0,n
    t=rho*math.sqrt((n-2)/(1-rho*rho)); p=2*(1-t_cdf(abs(t),n-2))
    return rho,p,n
def fisher_exact(a,b,c,d):
    # two-sided, table [[a,b],[c,d]]
    r1,r2,c1=a+b,c+d,a+c; n=r1+r2
    def pr(x): return math.comb(r1,x)*math.comb(r2,c1-x)/math.comb(n,c1)
    lo,hi=max(0,c1-r2),min(r1,c1); p0=pr(a)
    return min(1.0,sum(pr(x) for x in range(lo,hi+1) if pr(x)<=p0*(1+1e-7)))
def bh(p):
    p=np.asarray(p,float); q=np.full(len(p),np.nan); m=~np.isnan(p); pv=p[m]; n=len(pv)
    o=np.argsort(pv); r=pv[o]*n/np.arange(1,n+1); r=np.minimum.accumulate(r[::-1])[::-1]
    out=np.empty(n); out[o]=np.minimum(r,1); q[m]=out; return q
