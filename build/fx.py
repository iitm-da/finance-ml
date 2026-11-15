from lib import *

GREY = '#8FA0A9'; PALE = '#EEF2F4'

def jdata(id_, obj):
    return f'<script type="application/json" id="{id_}">{json.dumps(obj, separators=(",", ":"))}</script>'

def bld(n):
    return '<div class="bld">' + '<i></i>' * n + '</div>'

def G_(k, inner, to=None):
    t = f' data-to="{to}"' if to is not None else ''
    return f'<g data-at="{k}"{t}>{inner}</g>'

def tile(label, value, k=''):
    return f'<div class="lf {k}"><small>{label}</small>{value}</div>'

def box(x, y, w, h, title, sub='', col=INK, fill='#FFFFFF', tsize=13.5):
    s = Rect(x, y, w, h, fill, HAIR, 1) + Rect(x, y, w, 4, col)
    s += T(x + w / 2, y + (h / 2 + 3 if not sub else h / 2 - 3), title, tsize, INK, 'middle', BODY, 700)
    if sub:
        s += T(x + w / 2, y + h / 2 + 15, sub, 10.5, SOFT, 'middle', MONO)
    return s

class Ax:
    def __init__(self, w, h, xr, yr, pad=(46, 14, 14, 30)):
        self.w, self.h, self.xr, self.yr = w, h, xr, yr
        self.l, self.r, self.t, self.b = pad
        self.pw, self.ph = w - self.l - self.r, h - self.t - self.b

    def X(self, x):
        return self.l + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.pw

    def Y(self, y):
        return self.t + self.ph - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.ph

    def frame(self, yticks=(), xticks=(), yfmt=str, ylab='', xlab=''):
        s = Rect(self.l, self.t, self.pw, self.ph, '#FFFFFF', HAIR, 1)
        for v in yticks:
            s += L(self.l, self.Y(v), self.l + self.pw, self.Y(v), HAIR, 1)
            s += T(self.l - 6, self.Y(v) + 3.5, yfmt(v), 10, SOFT, 'end', MONO)
        for v, lab in xticks:
            s += T(self.X(v), self.t + self.ph + 14, lab, 10, SOFT, 'middle', MONO)
        if ylab:
            s += T(self.l + 6, self.t + 13, ylab, 10, SOFT, 'start', MONO)
        if xlab:
            s += T(self.l + self.pw, self.h - 5, xlab, 10, SOFT, 'end', MONO)
        return s

    def line(self, xs, ys, col, sw=2, op=1, dash='', cls=''):
        pts = ' '.join(f'{self.X(x):.1f},{self.Y(y):.1f}' for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ''
        c = f' class="{cls}"' if cls else ''
        return (f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-opacity="{op}" '
                f'stroke-linejoin="round"{d}{c}/>')

    def band(self, xs, lo, hi, col, op=.16):
        p = ' '.join(f'{self.X(x):.1f},{self.Y(y):.1f}' for x, y in zip(xs, hi))
        p += ' ' + ' '.join(f'{self.X(x):.1f},{self.Y(y):.1f}' for x, y in zip(reversed(xs), reversed(lo)))
        return f'<polygon points="{p}" fill="{col}" fill-opacity="{op}"/>'

    def vbar(self, x0, x1, y, col, base=None, op=1):
        base = self.yr[0] if base is None else base
        ya, yb = self.Y(max(y, base)), self.Y(min(y, base))
        return f'<rect x="{self.X(x0):.1f}" y="{ya:.1f}" width="{self.X(x1) - self.X(x0):.1f}" height="{yb - ya:.1f}" fill="{col}" fill-opacity="{op}"/>'

    def dot(self, x, y, col, r=3, extra=''):
        return f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{col}" {extra}/>'

def pct(v, d=1):
    return f'{v * 100:.{d}f}%'
