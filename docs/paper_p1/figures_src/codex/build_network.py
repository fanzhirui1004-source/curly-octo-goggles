"""Figure 6 (F11_network_architecture): network diagram. Imports no mechanics or model code.

Copied from 04_figures/architecture3/scripts/build_figures.py (function network(); the F01 overview
drawing of that script is not rebuilt here). Geometry of the drawing is unchanged; only visible labels
follow the R3 review (finding F198: symbols a, b; r_i, p_i; g_{lj}; W_{lh} as in Appendix G.1;
'Geometry branch', 'Indicators', 'Deployment'; 'network parameters' for theta).

Run from anywhere:  python3 docs/paper_p1/figures_src/codex/build_network.py
Writes docs/paper_p1/figures/F11_network_architecture.{png,pdf,svg} (PNG at 300 dpi).
"""
from pathlib import Path
from datetime import datetime, timezone
import os,json,io,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('P1_FIG_OUT',HERE.parents[1]/'figures'))
QA=HERE/'_build'/'qa'
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'_build'/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Ellipse, Polygon
from PIL import Image,ImageOps

BLUE='#0072B2'; INK='#243447'; GRAY='#657382'; LIGHT='#DFE5E9'; ORANGE='#D55E00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'mathtext.fontset':'stix',
                    'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'p1-architecture3'})
BUILD={}

class Drawing:
    def __init__(self,name,height,layout_height=None):
        self.name=name;self.h=height
        self.layout_h=layout_height or height
        self.yratio=self.layout_h/height
        self.fig=plt.figure(figsize=(180/25.4,height/25.4),facecolor='white')
        self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,180),ylim=(0,self.layout_h));self.ax.axis('off')
        self.texts=[];self.contained=[]
    def text(self,x,y,label,size=8.5,color=INK,**kw):
        assert size>=7.5
        t=self.ax.text(x,y,label,fontsize=size,color=color,**kw);self.texts.append(t);return t
    def line(self,pts,color=INK,dash=False,lw=.8,**kw):
        return self.ax.plot(*zip(*pts),color=color,lw=lw,ls=(0,(3,2)) if dash else '-',**kw)[0]
    def arrow(self,p,q,color=INK,dash=False,lw=.85,style='-|>',**kw):
        self.ax.add_patch(FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=8,lw=lw,
            color=color,linestyle=(0,(3,2)) if dash else '-',**kw))
    def box(self,x,y,w,h,label='',color=BLUE,size=8.5,fc=None,**kw):
        r=Rectangle((x,y),w,h,ec=color,fc=fc or color+'0D',lw=.75,**kw);self.ax.add_patch(r)
        if label:
            t=self.text(x+w/2,y+h/2,label,size,color=INK,ha='center',va='center');self.contained.append((r,t))
        return r
    def dot(self,x,y,r=1.1,color=BLUE,fc=None):
        self.ax.add_patch(Ellipse((x,y),2*r,2*r*self.yratio,ec=color,fc=fc or 'white',lw=.7))
    def plus(self,x,y,r=1.6,color=GRAY):
        self.dot(x,y,r,color,'white');self.line([(x-r*.6,y),(x+r*.6,y)],color,.0,.75)
        self.line([(x,y-r*.6*self.yratio),(x,y+r*.6*self.yratio)],color,.0,.75)
    def tensor(self,x,y,w=10,h=9,label='',color=BLUE):
        for k in [2,1,0]:
            self.ax.add_patch(Rectangle((x-w/2+k*.9,y-h/2+k*.8),w,h,ec=color,fc='white',lw=.65))
        self.ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,ec=color,fc=color+'14',lw=.65))
        for f in [.25,.5,.75]:
            self.line([(x-w/2+f*w,y-h/2),(x-w/2+f*w,y+h/2)],color,lw=.22)
        if label:self.text(x,y,label,8.5,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=.1))
    def pair(self,x,y,weak=False):
        self.box(x-8,y-7,16,14,color=BLUE,linestyle='--' if weak else '-')
        self.box(x-6.5,y-3,5.5,6,'E',BLUE,8,fc='white');self.box(x+1.5,y-3,5.5,6,'G',BLUE,8,fc='white')
        self.arrow((x-.7,y),(x+1.2,y),lw=.65)
        self.text(x,y-11,'×4',8.5,ha='center')
    def finish(self):
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();frame=self.fig.bbox
        outside=[];overflow=[]
        for t in self.texts:
            bb=t.get_window_extent(renderer)
            if bb.x0<-1 or bb.y0<-1 or bb.x1>frame.x1+1 or bb.y1>frame.y1+1:outside.append(t.get_text())
        for r,t in self.contained:
            rb=r.get_window_extent(renderer);tb=t.get_window_extent(renderer)
            if not(rb.contains(tb.x0,tb.y0) and rb.contains(tb.x1,tb.y1)):overflow.append(t.get_text())
        assert not outside,(self.name,'outside',outside)
        assert not overflow,(self.name,'overflow',overflow)
        for ext in ['svg','pdf','png']:
            metadata={'Date':'2026-09-27'} if ext=='svg' else ({'CreationDate':datetime(2026,9,27,tzinfo=timezone.utc)} if ext=='pdf' else {})
            out=io.BytesIO();self.fig.savefig(out,format=ext,dpi=300,facecolor='white',metadata=metadata)
            (OUT/f'{self.name}.{ext}').write_bytes(out.getvalue())
        with Image.open(OUT/f'{self.name}.png') as im:ImageOps.grayscale(im).save(QA/f'{self.name}_gray.png')
        tags=[x.tag.split('}')[-1] for x in ET.parse(OUT/f'{self.name}.svg').iter()];assert 'image' not in tags
        BUILD[self.name]={'width_mm':180,'height_mm':self.h,'minimum_font_pt':min(t.get_fontsize() for t in self.texts),
            'text_outside_canvas':outside,'text_overflow_boxes':overflow,'svg_text_nodes':tags.count('text'),'svg_raster_images':0,
            'no_mechanics_imports_or_execution':True}
        plt.close(self.fig)
        print(self.name,BUILD[self.name])


def network():
    # Compact vertical layout without scaling point-size typography or round nodes.
    d=Drawing('F11_network_architecture',220,layout_height=254)
    # (a) Actual encoder inputs and a bipartite interaction motif.
    d.text(7,245,'(a) Geometry branch',10,weight='bold')
    d.text(112,245,'Geometry-dependent coefficients',8.5,BLUE)
    d.text(7,234,'Element features',8)
    d.tensor(13,224,10,10,'126')
    d.text(7,215,'Moments / volume',7.5)
    d.text(7,207,'Node features',8)
    d.tensor(13,197,10,10,'11')
    d.text(7,187,'Indicators / stiffness / position',7.5)
    # Affine-GELU-affine encoders, displayed as neural layers rather than text boxes.
    for y in [224,197]:
        d.arrow((20, y),(28,y),BLUE)
        for xx in [30,37,44]:
            for yy in [y-3,y,y+3]:d.dot(xx,yy,.65,BLUE,BLUE+'18')
        for x1,x2 in [(30,37),(37,44)]:
            for y1 in [y-3,y,y+3]:
                for y2 in [y-3,y,y+3]:d.line([(x1+.7,y1),(x2-.7,y2)],BLUE,lw=.3)
        d.arrow((45,y),(51,y),BLUE)
        d.tensor(57,y,8,10,'64')
    d.text(37,231,'MLP',8,ha='center')
    # Element mean is an input to the initial node encoder, not a learned change of incidence.
    d.line([(57,218),(57,212),(34,212),(34,205),(25,205),(25,199)],BLUE,lw=.65)
    d.arrow((25,199),(25,197),BLUE,lw=.65)
    d.text(46,209,'mean',7.5,BLUE,ha='center')
    d.text(7,179,'MLP: affine → GELU → affine',8)
    d.arrow((64,224),(70,220),BLUE);d.arrow((64,197),(70,204),BLUE)
    for yy in [204,212,220]:
        d.box(72,yy-1.7,3.4,3.4,color=BLUE,fc=BLUE+'20');d.dot(94,yy,1.4,BLUE,BLUE+'20')
    for ya in [204,212,220]:
        for yb in [204,212,220]:d.line([(75.5,ya),(92.5,yb)],BLUE,lw=.4)
    d.arrow((78,223),(90,223),BLUE,style='<->',lw=.8)
    d.text(83,230,'64-channel exchange',8,ha='center')
    d.text(83,225,'2 residual rounds',7.5,ha='center')
    d.text(74,199,'E',8,ha='center');d.text(94,199,'N',8,ha='center')
    # Local heads use element/face + node + learned slot embeddings.
    d.arrow((97,216),(120,224),BLUE)
    d.tensor(110,208,7,7,'64')
    d.text(110,214,'Face',7.5,ha='center')
    d.arrow((77,204),(105,208),BLUE,lw=.65)
    d.arrow((116,209),(121,218),BLUE,lw=.65)
    d.tensor(111,237,6,5,'8')
    d.text(121,237,'slot embedding',7.5,va='center')
    d.arrow((113,233),(128,230),BLUE)
    d.box(122,216,49,14,'Local heads  →  '+r'$a,b$'+'\nMLP + bounded coefficients',BLUE,8)
    # Multilevel node embeddings supply geometry-only transfer and convolution heads.
    d.line([(96,202),(99,202),(99,194)],BLUE);d.arrow((99,194),(109,194),BLUE)
    d.tensor(113,194,7,8,'64')
    d.text(97,179,'Node hierarchy',7.5)
    d.box(124,190,47,14,r'Transfer heads  →  $r_i,p_i$'+'\nPositive geometry coefficients',BLUE,8)
    d.arrow((119,197),(123,197),BLUE)
    d.box(124,172,47,12,r'Convolution heads  →  $g_{\ell j}$'+'\n'+r'$2\,\mathrm{sigmoid}$',BLUE,8)
    d.line([(118,190),(121,190),(121,178)],BLUE);d.arrow((121,178),(123,178),BLUE)
    # Dashed coefficient buses enter only the numerical maps of the q branch.
    d.line([(172,223),(176,223),(176,162),(56,162)],BLUE,True)
    for xx in [56,88,106,130]:d.arrow((xx,162),(xx,154),BLUE,True)
    d.line([(172,197),(178,197),(178,162)],BLUE,True)
    d.line([(172,178),(177,178),(177,162)],BLUE,True)

    # (b) The actual MGNO2 forward sequence and two deterministic bypasses.
    d.text(7,166,'(b) Displacement branch: linear in '+r'$q$'+' for fixed geometry',10,weight='bold')
    y=146
    d.text(7,y,r'$q$',12,ha='center',va='center')
    d.box(16,139,15,14,r'$\Pi_P$',GRAY,12)
    d.tensor(43,y,10,12,r'$X^0$')
    d.text(43,135,'3 → 32',8,ha='center')
    d.pair(64,y);d.text(64,155,'Local',8,ha='center')
    d.box(80,139,17,14,color=BLUE)
    # Four levels and a return branch are a miniature of panel c.
    pp=[(82,149),(84,147),(86,145),(88,143),(90,143),(92,145),(94,147),(96,149)]
    d.line(pp,BLUE,lw=.8)
    for x,z in pp:d.dot(x,z,.55,BLUE,BLUE+'20')
    d.text(88,135,'3 levels',8,ha='center')
    d.pair(114,y);d.text(114,155,'Local',8,ha='center')
    d.pair(138,y,True);d.text(138,155,'Weak',8,ha='center')
    d.tensor(155,y,4,10,'')
    d.text(155,155,'32 → 3',8,ha='center')
    d.plus(163,y,1.6)
    d.box(169,141,7,10,r'$P$',GRAY,10)
    d.text(173,155,r'$\widehat E q$',10,ha='center')
    d.arrow((172.5,151),(172.5,154),GRAY)
    for x1,x2 in [(10,16),(31,37),(50,55),(73,80),(98,105),(123,129),(147,151),(158,161),(165,169)]:d.arrow((x1,y),(x2,y))
    d.line([(23.5,139),(23.5,130),(163,130),(163,144)],GRAY,lw=.8)
    d.arrow((163,138),(163,144),GRAY)
    d.text(83,131,r'Rigid field $RC_Rq$',8,GRAY,ha='center',va='bottom')
    d.line([(7,142),(7,123),(172.5,123),(172.5,140)],GRAY,lw=.8)
    d.arrow((172.5,137),(172.5,140),GRAY)
    d.text(84,124,r'Prescribed $q$ restored on $P$',8,GRAY,ha='center',va='bottom')

    # (c) Exact down/up order, with additive skip states and no channel concatenation.
    d.text(7,114,'(c) Multiscale displacement propagation',9.5,weight='bold')
    xL,xR=25,77; ys=[104,83,62,41]
    d.text(49,108,'32 channels',8,BLUE,ha='center')
    for i,(m,yy) in enumerate(zip([65,33,17,9],ys)):
        size=[12,11,10,9][i]
        d.text(7,yy,str(m),9,ha='center',va='center')
        d.tensor(xL,yy,size,9,r'$X$' if i==0 else '2 × C')
        d.tensor(xR,yy,size,9,r'$X$' if i==0 else '2 × C')
        if i<3:
            d.arrow((xL,yy-5),(xL,ys[i+1]+6),BLUE)
            d.text(15,yy-12,rf'$\mathcal{{R}}_{i}$',9,BLUE,ha='center')
            py=yy-10
            d.plus(xR,py,1.6)
            d.arrow((xR,ys[i+1]+6),(xR,py-1.7),BLUE)
            d.arrow((xR,py+1.7),(xR,yy-4.8),BLUE)
            d.text(89,yy-14,rf'$\sigma_{i}\mathcal{{I}}_{i}$',9,BLUE,ha='center')
            d.line([(xL+size/2+2,yy),(55,yy),(55,py),(xR-1.7,py)],GRAY,lw=.8)
            d.arrow((69,py),(xR-1.7,py),GRAY)
            if i==1:d.text(55,yy+3,'skip',8,GRAY,ha='center')
    d.arrow((xL+7,41),(xR-6,41),BLUE)
    d.text(51,45,'down ×2 → up ×2',8,ha='center')
    d.text(7,29,r'$\mathcal{R}$: restriction   $\mathcal{I}$: prolongation',8)
    d.text(7,23,r'$C:\ X\leftarrow X+g\odot\mathrm{Conv}_{3\times3\times3}(X)$',9)
    d.box(89,100,8,8,r'$P$',GRAY,10)
    d.arrow((85,104),(89,104),GRAY)

    # (d) Enlarged one-interaction map: slots -> 4 heads -> channel matrices -> slots.
    d.text(106,114,'(d) Local interaction',9.5,weight='bold')
    d.text(110,106,'27 slots',8,ha='center');d.text(146,106,'4 heads',8,ha='center');d.text(173,106,'nodes',8,ha='center')
    in_y=[98,90,82,74,66];head_y=[96,87,78,69]
    for yy in in_y:d.dot(110,yy,1.15,BLUE,BLUE+'1C');d.dot(173,yy,1.15,BLUE,BLUE+'1C')
    for h,yy in enumerate(head_y):
        for iy in in_y:d.line([(111.2,iy),(135,yy)],BLUE,lw=.28)
        d.dot(137,yy,1.45,BLUE,BLUE+'18')
        d.arrow((139,yy),(144,yy),lw=.65)
        d.box(145,yy-2.5,7,5,color=BLUE,fc=BLUE+'15')
        for f in [1/3,2/3]:
            d.line([(145+7*f,yy-2.5),(145+7*f,yy+2.5)],BLUE,lw=.3)
            d.line([(145,yy-2.5+5*f),(152,yy-2.5+5*f)],BLUE,lw=.3)
        for iy in in_y:d.line([(152.4,yy),(171.7,iy)],BLUE,lw=.28)
    d.text(122,101,r'$a$',10,BLUE);d.text(160,101,r'$b$',10,BLUE)
    d.text(148.5,60,r'$W_{\ell h}:32\times32$',9,ha='center')
    d.line([(110,64),(110,54),(167,54)],GRAY)
    d.plus(173,54,1.5);d.arrow((167,54),(171.5,54),GRAY)
    d.arrow((173,64),(173,55.7))
    d.box(121,39,43,11,r'Restore $X^0$ on $P$',GRAY,8.5)
    d.line([(173,52.4),(173,44.5),(165,44.5)],GRAY);d.arrow((168,44.5),(164.5,44.5),GRAY)
    d.text(106,30,'E: 27 Q2 nodes',8)
    d.text(106,24,'G: 18 owner + 9 neighbour nodes',8)
    d.line([(7,17),(174,17)],LIGHT,lw=.65)
    d.text(7,8,r'Training: update network parameters $\theta$ through both branches.',8)
    d.text(100,8,r'Deployment: fix $\theta$; cache geometry coefficients.',8)
    d.finish()


if __name__=='__main__':
    for q in [OUT,QA]:q.mkdir(parents=True,exist_ok=True)
    network()
    (QA/'BUILD_CHECKS_network.json').write_text(json.dumps(BUILD,indent=2)+'\n')
