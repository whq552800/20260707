import numpy as np
def ns_basis(x, knots, bk=(0.0,1.0)):
    k=np.r_[bk[0],knots,bk[1]]; K=len(k)
    def dk(j): return (np.clip(x-k[j],0,None)**3-np.clip(x-k[-1],0,None)**3)/(k[-1]-k[j])
    return np.column_stack([x]+[dk(j)-dk(K-2) for j in range(K-2)])
def period_of(y):
    return '1990–2000' if y<=2000 else ('2001–2010' if y<=2010 else '2011–2023')
