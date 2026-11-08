"""Independent, documented *new* quasi-steady Fiala bicycle-model scenario study.
Not a reproduction of the unavailable historical nonlinear solver.
Usage: python article4_reproducible_fiala.py --input real_vs_Test.csv --output results.csv [--limit 100]
Requires numpy, pandas, scipy. Vehicle properties are explicit HYPOTHETICAL assumptions.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import root

G=9.81
# Assumed scenario configurations; not measured Buick/production vehicle parameters.
VEHICLES={
 'compact_assumed':dict(L=2.650,ratio=14.,m=1450.,lf_fraction=.44,Cf=80000.,Cr=85000.),
 'mid_assumed':dict(L=3.088,ratio=16.,m=1800.,lf_fraction=.44,Cf=90000.,Cr=105000.),
 'large_assumed':dict(L=3.400,ratio=18.,m=2200.,lf_fraction=.44,Cf=110000.,Cr=125000.),
}

def fiala(alpha,C,mu,Fz):
    # positive alpha -> positive lateral force, with saturation at +/-mu*Fz
    z=np.tan(np.clip(alpha,-1.2,1.2))
    q=np.minimum(np.abs(C*z)/(3*mu*Fz),1.)
    return np.sign(z)*mu*Fz*(1-(1-q)**3)

def solve(v,sw,mu,p):
    if not np.isfinite(v) or not np.isfinite(sw):return dict(status='invalid_input')
    if v<2.:return dict(status='low_speed')
    L,ratio,m,Cf,Cr=(p[k] for k in ('L','ratio','m','Cf','Cr'))
    lf=L*p['lf_fraction'];lr=L-lf
    Fzf=m*G*lr/L;Fzr=m*G*lf/L
    delta=np.deg2rad(sw)/ratio
    rk=v*np.tan(delta)/L
    def calc(x):
        beta,r=x
        af=delta-np.arctan(beta+lf*r/v)
        ar=-np.arctan(beta-lr*r/v)
        ff=fiala(af,Cf,mu,Fzf);fr=fiala(ar,Cr,mu,Fzr)
        # force equilibrium and yaw-moment equilibrium, normalized by force scale
        return np.array([(ff*np.cos(delta)+fr-m*v*r)/(m*G),
                         (lf*ff*np.cos(delta)-lr*fr)/(m*G*L)])
    candidates=[]
    for b0,r0 in ((0.,rk),(0.,0.),(.05,rk),(-.05,rk)):
        s=root(calc,[b0,r0],tol=1e-9)
        if not np.all(np.isfinite(s.x)):continue
        b,r=s.x;err=float(np.max(np.abs(calc(s.x))))
        if err<1e-7 and abs(b)<=.3 and abs(r)<=1.5:
            af=delta-np.arctan(b+lf*r/v);ar=-np.arctan(b-lr*r/v)
            uf=abs(fiala(af,Cf,mu,Fzf))/(mu*Fzf)
            ur=abs(fiala(ar,Cr,mu,Fzr))/(mu*Fzr)
            candidates.append((abs(r-rk),b,r,max(uf,ur),err))
    if not candidates:return dict(status='no_verified_root')
    _,b,r,u,err=min(candidates,key=lambda z:z[0])
    return dict(status='ok',beta=b,nonlinear_yaw=r,max_axle_util=u,residual=err,
                kinematic_yaw=rk,yaw_relative_deviation=abs(r-rk)/max(abs(rk),.01))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='real_vs_Test.csv')
    ap.add_argument('--output',default='Article4_new_fiala_results.csv')
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    d=pd.read_csv(a.input)
    required={'route_id','frame_index','real_steering_deg','real_speed_kmh','test_steering_deg','test_speed_kmh'}
    assert required.issubset(d.columns),f'Missing {required-set(d.columns)}'
    assert len(d)==3000, 'Expected 3000 paired samples; check data provenance'
    if a.limit:d=d.iloc[:a.limit]
    rows=[]
    for vehicle,p in VEHICLES.items():
        for mu in (.1,.3):
            for side,sp,st in [('real','real_speed_kmh','real_steering_deg'),('test','test_speed_kmh','test_steering_deg')]:
                for z in d.itertuples(index=False):
                    v=getattr(z,sp)/3.6;sw=getattr(z,st)
                    r=solve(v,sw,mu,p)
                    rows.append(dict(vehicle=vehicle,mu=mu,side=side,route_id=z.route_id,
                                     frame_index=z.frame_index,v_ms=v,sw_deg=sw,**r))
    out=pd.DataFrame(rows)
    out.to_csv(a.output,index=False,float_format='%.10g')
    print('Wrote',a.output,'rows=',len(out))
    print(out.groupby(['vehicle','mu','side','status']).size().to_string())
if __name__=='__main__':main()