import ezdxf,sys,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy
fn,out=sys.argv[1],sys.argv[2]
win=[float(x) for x in sys.argv[3:7]] if len(sys.argv)>3 else None
d=ezdxf.readfile(fn)
fig=plt.figure(figsize=(16,16),dpi=110); ax=fig.add_axes([0,0,1,1])
Frontend(RenderContext(d),MatplotlibBackend(ax),config=Configuration(background_policy=BackgroundPolicy.WHITE)).draw_layout(d.modelspace())
if win: ax.set_xlim(win[0],win[2]); ax.set_ylim(win[1],win[3])
fig.savefig(out)
