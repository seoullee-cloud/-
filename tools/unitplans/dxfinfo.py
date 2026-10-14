import ezdxf,sys,collections
from ezdxf import bbox
for fn in sys.argv[1:]:
    d=ezdxf.readfile(fn); m=d.modelspace()
    print('=====',fn,d.dxfversion,'units',d.header.get('$INSUNITS'))
    c=collections.Counter((e.dxf.layer,e.dxftype()) for e in m)
    for k,v in sorted(c.items()): print('  ',k,v)
    ins=collections.Counter(e.dxf.name for e in m.query('INSERT'))
    print('  inserts',dict(ins))
    ext=bbox.extents(m); print('  ext',ext.extmin,ext.extmax)
