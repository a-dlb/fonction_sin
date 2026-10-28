"""
Cercle trigonométrique et fonction sinus : g(x) = A·sin(ω·x + φ) + B
Version Streamlit.

Utilisation :  streamlit run cercle_sinus_streamlit.py
Dépendances :  pip install streamlit numpy matplotlib
"""
import time

import numpy as np
import streamlit as st
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

rcParams["mathtext.fontset"] = "cm"

# --------------------------------------------------------------- paramètres
DEFAULTS = {"A": 1.0, "w": 1.0, "phi": 0.0, "B": 0.0, "x0": 0.0}
X_MIN = -2 * np.pi
X_MAX = 4 * np.pi
LIM_MIN = 2

COL_A = "#e6b800"
COL_W = "#2e86c1"
COL_PHI = "#28b463"
COL_B = "#e74c3c"
COL_TH = "#8e44ad"
COL_A_TXT = "#b38f00"
COL_X = "#7f8c8d"

X = np.linspace(X_MIN, X_MAX, 1000)
T_CIRCLE = np.linspace(0, 2 * np.pi, 400)


# ------------------------------------------------------------------ outils
def sign(v):
    return r"$-$" if round(v, 2) < 0 else r"$+$"


def draw_arc(ax, color, cx, cy, r, a0, a1, label, lw=2.5):
    """Arc orienté (avec pointe de flèche) + étiquette."""
    span = a1 - a0
    if abs(span) < 0.03:
        return
    n = max(int(abs(span) / 0.05) + 2, 4)
    ang = np.linspace(a0, a1, n)
    ax.plot(cx + r * np.cos(ang), cy + r * np.sin(ang), color=color, lw=lw,
            zorder=3)
    tail = a1 - np.sign(span) * min(0.15, abs(span) / 2)
    ax.annotate(
        "", xy=(cx + r * np.cos(a1), cy + r * np.sin(a1)),
        xytext=(cx + r * np.cos(tail), cy + r * np.sin(tail)),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0,
                        shrinkB=0, mutation_scale=14),
        zorder=3)
    am = a1 - np.sign(span) * min(0.35, abs(span) / 2)
    rt = r + 0.32
    ax.text(cx + rt * np.cos(am), cy + rt * np.sin(am), label, color=color,
            fontsize=11, zorder=6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.8))


def point_label(ax, pos, text, color):
    ax.text(pos[0], pos[1], text, color=color, fontsize=11, zorder=6,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.8))


def arrow(ax, color, xy, xytext, style="<->"):
    ax.annotate("", xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle=style, color=color, lw=2,
                                shrinkA=0, shrinkB=0, mutation_scale=14),
                zorder=4)


def label(ax, pos, text, color, **kw):
    ax.text(pos[0], pos[1], text, color=color, fontsize=12, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.75), **kw)


# ------------------------------------------------------- annotations courbe
def annotate_curve(ax_g, A, w, phi, B, lim, show):
    T = 2 * np.pi / w
    x_ph = -phi / w

    if show["B"] and abs(B) >= 0.05:
        xB = X_MIN + 0.35
        arrow(ax_g, COL_B, (xB, B), (xB, 0))
        label(ax_g, (xB + 0.15, B / 2), rf"$B={B:.1f}$", COL_B,
              ha="left", va="center")

    if A < 0.05:
        return

    # φ : montée par l'axe médian
    if show["φ"]:
        in_view = X_MIN <= x_ph <= X_MAX
        x_vis = x_ph if in_view else X_MIN + (x_ph - X_MIN) % T
        if x_vis <= X_MAX:
            ax_g.plot([x_vis], [B], "o", color=COL_PHI, mec="k", ms=10,
                      zorder=6)
            txt = rf"$-\varphi/\omega={x_ph:.2f}$"
            if not in_view:
                txt += "\n(à $kT$ près)"
            label(ax_g, (x_vis + 0.12, B - 0.15), txt, COL_PHI,
                  ha="left", va="top")
        if in_view and abs(x_ph) > 0.05:
            arrow(ax_g, COL_PHI, (x_ph, B), (0, B), style="->")

    # A et T : sur un maximum visible
    lo = X_MIN + 0.8
    x_pk = lo + (x_ph + T / 4 - lo) % T
    top = B + A
    if x_pk <= X_MAX and top <= lim:
        if show["A"]:
            arrow(ax_g, COL_A, (x_pk, top), (x_pk, B))
            label(ax_g, (x_pk + 0.12, B + A / 2), rf"$A={A:.1f}$", COL_A_TXT,
                  ha="left", va="center")
        if show["ω"] and x_pk + T <= X_MAX:
            arrow(ax_g, COL_W, (x_pk + T, top), (x_pk, top))
            label(ax_g, (x_pk + T / 2, top - 0.12),
                  rf"$T=2\pi/\omega={T:.2f}$", COL_W, ha="center", va="top")


# ------------------------------------------------------- angles du cercle
def annotate_circle(fig, ax_c, A, w, phi, B, x0):
    cx, cy = 0.0, B
    rb = max(A, 1.0)
    wx = w * x0
    theta = wx + phi
    show_w = abs(w - 1.0) > 0.005
    phi_zero = abs(phi) < 0.005
    coincide_x = (not show_w) and phi_zero

    def pt(angle, r):
        return cx + r * np.cos(angle), cy + r * np.sin(angle)

    # 1) x (gris)
    px, py = pt(x0, A)
    ax_c.plot([cx, px], [cy, py], color=COL_X, lw=2, zorder=3)
    if not coincide_x:
        ax_c.plot([px], [py], "o", color=COL_X, ms=9, zorder=4)
        point_label(ax_c, pt(x0, A + 0.35), r"$x$", COL_X)

    # 2) x -> ω·x (bleu)
    if show_w:
        draw_arc(ax_c, COL_W, cx, cy, 0.55 * rb, x0, wx, r"$\times\omega$")
        if not phi_zero:
            qx, qy = pt(wx, A)
            ax_c.plot([cx, qx], [cy, qy], "--", color=COL_W, lw=1.5,
                      zorder=3)
            ax_c.plot([qx], [qy], "o", color=COL_W, ms=5, zorder=4)
            point_label(ax_c, pt(wx, A + 0.35), r"$\omega x$", COL_W)

    # 3) ω·x -> θ (vert)
    draw_arc(ax_c, COL_PHI, cx, cy, 0.70 * rb, wx, theta, r"$+\varphi$")

    # 4) θ
    point_label(ax_c, pt(theta, A + 0.35), r"$\theta$", COL_TH)

    # 5) légende au-dessus du cercle
    LEG_X, LEG_Y, LEG_W, LEG_ROW = 0.01, 0.78, 0.28, 0.030
    rows = [(COL_X, rf"$x = {x0:.2f}$ rad")]
    if show_w and not phi_zero:
        rows.append((COL_W,
                     rf"$\omega x = {w:.1f}\times{x0:.2f} = {wx:.2f}$ rad"))
    rows.append((COL_PHI, rf"$\varphi = {phi:+.2f}$ rad  (phase à l'origine)"))
    rows.append((COL_TH, rf"$\theta = \omega x + \varphi = {theta:.2f}$ rad"))
    h = LEG_ROW * len(rows) + 0.012
    fig.add_artist(Rectangle((LEG_X, LEG_Y), LEG_W, h,
                             transform=fig.transFigure, fc="white",
                             ec="0.7", alpha=0.9, zorder=6))
    for i, (col, txt) in enumerate(rows):
        fig.text(LEG_X + 0.012, LEG_Y + h - 0.021 - LEG_ROW * i, txt,
                 fontsize=10.5, color=col, ha="left", va="center", zorder=7)


# ------------------------------------------------------------------- titre
def draw_title(fig, A, w, phi, B):
    FS, FS_LET = 16, 14
    Y_FORM, Y_LET = 0.945, 0.970
    Y_UNDER = Y_FORM - 0.034
    Y_THETA = Y_UNDER - 0.005
    r = fig.canvas.get_renderer()

    def boxed(color, txt):
        return fig.text(0, Y_FORM, txt, fontsize=FS, ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc=to_rgba(color, 0.25), ec=color, lw=2))

    def plain(txt):
        return fig.text(0, Y_FORM, txt, fontsize=FS, ha="left", va="center")

    def letter(tex, color):
        return fig.text(0, Y_LET, tex, fontsize=FS_LET, color=color,
                        ha="center", va="bottom")

    t_lp, t_rp = plain(r"$($"), plain(r"$)$")
    items = [
        (plain(r"$g(x)=$"), None, False, 1),
        (boxed(COL_A, rf"${A:.1f}$"), letter(r"$A$", COL_A), True, 1),
        (plain(r"$\sin$"), None, False, 0),
        (t_lp, None, False, 0.4),
        (boxed(COL_W, rf"${w:.1f}$"), letter(r"$\omega$", COL_W), True, 1),
        (plain(r"$x$"), None, False, 1),
        (plain(sign(phi)), None, False, 1),
        (boxed(COL_PHI, rf"${abs(phi):.2f}$"), letter(r"$\varphi$", COL_PHI),
         True, 1),
        (t_rp, None, False, 1),
        (plain(sign(B)), None, False, 1),
        (boxed(COL_B, rf"${abs(B):.1f}$"), letter(r"$B$", COL_B), True, 0),
    ]

    W = fig.bbox.width
    pad_px = 0.3 * FS * fig.dpi / 72
    gap = 0.30 * FS * fig.dpi / 72
    widths = [a.get_window_extent(r).width + (2 * pad_px if b else 0)
              for a, _, b, _ in items]
    total = sum(wd + gap * g for wd, (_, _, _, g) in zip(widths, items))
    xpx = 0.5 * W - total / 2

    start = {}
    for (art, let, b, g), wd in zip(items, widths):
        art.set_position(((xpx + (pad_px if b else 0)) / W, Y_FORM))
        if let is not None:
            let.set_position(((xpx + wd / 2) / W, Y_LET))
        start[art] = (xpx, wd)
        xpx += wd + gap * g

    x0 = start[t_lp][0]
    x1 = start[t_rp][0] + start[t_rp][1]
    fig.add_artist(Line2D([x0 / W, x1 / W], [Y_UNDER, Y_UNDER],
                          color=COL_TH, lw=2.5, transform=fig.transFigure,
                          solid_capstyle="round"))
    fig.text(0.5 * (x0 + x1) / W, Y_THETA, r"$\theta$", fontsize=FS_LET,
             color=COL_TH, ha="center", va="top")


def draw_subtitle(fig, ax_g, x0, w, theta):
    FS_SUB, Y_SUB = 12, 0.715
    vals = [(r"$x$", COL_X), (rf"$= {x0:.2f}$", "k"), (r"$\to$", "k"),
            (r"$\omega x$", COL_W), (rf"$= {w * x0:.2f}$", "k"),
            (r"$\to$", "k"),
            (r"$\theta = \omega x + \varphi$", COL_TH),
            (rf"$= {theta:.2f}\ \mathrm{{rad}}$", "k")]
    texts = [fig.text(0, Y_SUB, v, fontsize=FS_SUB, color=c, ha="left",
                      va="baseline") for v, c in vals]
    r = fig.canvas.get_renderer()
    W = fig.bbox.width
    gap = 0.25 * FS_SUB * fig.dpi / 72
    widths = [t.get_window_extent(r).width for t in texts]
    total = sum(widths) + gap * (len(texts) - 1)
    pos = ax_g.get_position()
    xpx = (pos.x0 + pos.width / 2) * W - total / 2
    for t, wd in zip(texts, widths):
        t.set_position((xpx / W, Y_SUB))
        xpx += wd + gap


# ------------------------------------------------------------------ figure
def build_figure(A, w, phi, B, x0, show):
    fig = Figure(figsize=(12, 7.5), dpi=100)
    FigureCanvasAgg(fig)

    lim = max(LIM_MIN, int(np.floor(round(A + abs(B), 6))) + 1)
    ax_c = fig.add_axes([0.05, 0.08, 0.31, 0.62])
    ax_g = fig.add_axes([0.43, 0.08, 0.54, 0.62], sharey=ax_c)

    theta = w * x0 + phi
    px, py = A * np.cos(theta), B + A * np.sin(theta)

    # Cercle
    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.axhline(0, color="gray", lw=0.8)
    ax_c.axvline(0, color="gray", lw=0.8)
    ax_c.set_title("Cercle trigonométrique")
    ax_c.grid(alpha=0.3)
    ax_c.plot(A * np.cos(T_CIRCLE), B + A * np.sin(T_CIRCLE), color="black")
    ax_c.plot([0], [B], "k+", ms=10)
    ax_c.plot([0, px], [B, py], color=COL_TH, lw=2, zorder=4)
    ax_c.plot([px], [py], "o", color=COL_TH, ms=9, zorder=5)

    # Courbe
    ax_g.set_xlim(X_MIN, X_MAX)
    ax_g.axhline(0, color="gray", lw=0.8)
    ax_g.axvline(0, color="gray", lw=0.8)
    ks = np.arange(-2, 5)
    ax_g.set_xticks(ks * np.pi)
    ax_g.set_xticklabels(
        [r"$0$" if k == 0 else
         (r"$\pi$" if k == 1 else (r"$-\pi$" if k == -1 else rf"${k}\pi$"))
         for k in ks])
    ax_g.set_xlabel(r"$x$")
    ax_g.grid(alpha=0.3)
    ax_g.plot(X, A * np.sin(w * X + phi) + B, color="tab:blue", lw=2)
    ax_g.plot([X_MIN, X_MAX], [B, B], "--", color=COL_B, lw=1.2,
              label="axe médian y = B")
    ax_g.plot([x0], [py], "o", color=COL_TH, ms=9, zorder=5)
    ax_g.legend(loc="upper right")
    ax_g.tick_params(labelleft=False)

    fig.add_artist(ConnectionPatch((px, py), (x0, py), "data", "data",
                                   axesA=ax_c, axesB=ax_g,
                                   color=COL_TH, ls="--", lw=1))

    annotate_curve(ax_g, A, w, phi, B, lim, show)
    if show["x"]:
        annotate_circle(fig, ax_c, A, w, phi, B, x0)

    draw_title(fig, A, w, phi, B)
    draw_subtitle(fig, ax_g, x0, w, theta)
    return fig


# --------------------------------------------------------------- interface
def reset_params():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["play"] = False


def main():
    st.set_page_config(page_title="Cercle trigonométrique et sinus",
                       layout="wide")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("play", False)

    sb = st.sidebar
    sb.header("Paramètres")
    A = sb.slider("A (amplitude)", 0.0, 4.0, step=0.1, key="A")
    w = sb.slider("ω (pulsation)", 0.1, 4.0, step=0.1, key="w")
    phi = sb.slider("φ (phase à l'origine)", -3.15, 3.15, step=0.05,
                    key="phi")
    B = sb.slider("B (décalage)", -2.0, 2.0, step=0.1, key="B")
    x0 = sb.slider("x (position)", 0.0, float(X_MAX), step=0.05, key="x0")

    c1, c2 = sb.columns(2)
    c1.button("Réinitialiser", on_click=reset_params,
              use_container_width=True)
    play = c2.toggle("▶ Lecture", key="play")

    sb.header("Annotations")
    show = {
        "A": sb.checkbox("A (amplitude)", True),
        "ω": sb.checkbox("ω (période T)", True),
        "φ": sb.checkbox("−φ/ω (déphasage)", True),
        "B": sb.checkbox("B (décalage)", True),
        "x": sb.checkbox("Angles x, ωx, φ, θ (cercle)", True),
    }

    placeholder = st.empty()

    def render(xv):
        fig = build_figure(A, w, phi, B, xv, show)
        placeholder.pyplot(fig, clear_figure=True)

    render(x0)

    # Animation : la boucle s'interrompt dès que l'utilisateur touche un widget
    if play:
        xv = x0
        while True:
            xv = (xv + 0.15) % X_MAX
            render(xv)
            time.sleep(0.02)


if __name__ == "__main__":
    main()