# Distribution of retained box area fraction v of cells cut by a straight vertical trimming plane
# n = (cos t, sin t), keep n.x <= b, over a large grid; t uniform in [0,45deg], b uniform.
import numpy as np
rng=np.random.default_rng(0)
def frac_below(nx,ny,b,x0,y0,m=64):
    # fraction of unit square [x0,x0+1]x[y0,y0+1] with nx*x+ny*y<=b (sampled)
    s=(np.arange(m)+.5)/m
    X,Y=np.meshgrid(s,s,indexing='ij')
    return ((nx*(x0[:,None,None]+X)+ny*(y0[:,None,None]+Y))<=b).mean((1,2))
allv=[]; per_len=[]
for trial in range(400):
    t=rng.uniform(0,np.pi/4); nx,ny=np.cos(t),np.sin(t); b=rng.uniform(0,1)+20
    I,J=np.meshgrid(np.arange(0,40),np.arange(0,40),indexing='ij'); I=I.ravel();J=J.ravel()
    # cells intersecting the line
    c=nx*I+ny*J; lo=c; hi=c+nx+ny
    cut=(lo<b)&(hi>b)
    # restrict to a window of line length ~ 10 cells around the centre
    v=frac_below(nx,ny,b,I[cut].astype(float),J[cut].astype(float))
    allv.append(v); per_len.append(cut.sum()/ (40/ max(np.sin(t),1e-9) if False else 1))
v=np.concatenate(allv)
print('cut cells sampled',len(v))
for lo,hi in [(0,0.02),(0.02,0.10),(0.10,0.34),(0.34,0.66),(0.66,0.98),(0.98,1.0)]:
    print('v in [%.2f,%.2f): %.3f'%(lo,hi,((v>=lo)&(v<hi)).mean()))
print('mean cut cells per unit boundary length = cos t + sin t, avg over t in [0,45]:', np.mean([np.cos(t)+np.sin(t) for t in np.linspace(0,np.pi/4,1001)]))
