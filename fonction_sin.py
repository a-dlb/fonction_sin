"""
Cercle trigonométrique et fonction sinus : g(x) = A·sin(ω·x + φ) + B

- Gauche : un point parcourt le cercle (noir) de centre (0, B) et de rayon |A|,
           avec l'angle θ = ω·x + φ. Les étapes qui mènent à θ sont dessinées :
           * x      (gris)  : rayon et point sur le cercle pour l'angle x
           * ω·x    (bleu)  : rayon pointillé et petit point ; la flèche
                              x -> ω·x montre l'effet de la pulsation
                              (masqué quand ω = 1)
           * +φ     (vert)  : flèche ω·x -> θ, la phase à l'origine
           * θ      (mauve) : angle final, qui donne le point du cercle
           Une légende (au-dessus du cercle) donne les valeurs de chaque
           angle.
           Quand φ = 0 (θ = ω·x), le point et l'étiquette de ω·x sont masqués
           pour ne pas se superposer à ceux de θ.
- Droite : la courbe g(x). L'ordonnée du point sur le cercle est
           exactement g(x) : le trait pointillé (mauve) le montre.
- Titre  : chaque paramètre est encadré dans sa couleur, avec sa lettre
           au-dessus ; la parenthèse de sin( ) est soulignée en mauve et
           étiquetée θ.
- Axes y : limites ±(⌊A + |B|⌋ + 1), au minimum ±2 ; elles ne changent que
           lorsque A + |B| franchit un entier (A = 2 → ±3, etc.).
- Sous-titre : x (gris), ωx (bleu) et θ = ωx + φ (mauve) sont dans leur
           couleur ; les valeurs numériques et les flèches restent noires.
- Courbe : les paramètres sont aussi visualisés sur le graphique :
           * B  (rouge)  : décalage vertical de l'axe médian y = B
           * A  (jaune)  : amplitude, de l'axe médian jusqu'au maximum
           * T  (bleu)   : période T = 2π/ω, entre deux maxima
           * -φ/ω (vert) : déphasage, point où la courbe coupe l'axe médian en
                           montant (φ est la phase à l'origine)
  Des cases à cocher permettent de choisir quelles annotations afficher.

Utilisation : python cercle_sinus_v7.py   (nécessite numpy et matplotlib)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import ConnectionPatch, Rectangle

plt.rcParams.update({
    "mathtext.fontset": "cm",
})

# --------------------------------------------------------------- paramètres
A0, W0, PHI0, B0, X0 = 1.0, 1.0, 0.0, 0.0, 0.0
X_MIN = -2 * np.pi       # on montre aussi les x négatifs (pour -φ/ω > 0)
X_MAX = 4 * np.pi
LIM_MIN = 2              # limite minimale (identique) des axes y

# Couleur associée à chaque paramètre (cadre du titre + curseur + annotations)
COL_A = "#e6b800"        # jaune
COL_W = "#2e86c1"        # bleu
COL_PHI = "#28b463"      # vert
COL_B = "#e74c3c"        # rouge
COL_TH = "#8e44ad"       # mauve : θ
COL_A_TXT = "#b38f00"    # jaune plus foncé pour que le texte reste lisible
COL_X = "#7f8c8d"        # gris : l'angle x

x = np.linspace(X_MIN, X_MAX, 1000)
t = np.linspace(0, 2 * np.pi, 400)

# état des limites des axes y (changent par paliers d'une unité)
lim_state = {"v": LIM_MIN}

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(12, 7))
ax_c = fig.add_axes([0.05, 0.34, 0.31, 0.46])
ax_g = fig.add_axes([0.43, 0.34, 0.54, 0.46], sharey=ax_c)

# Cercle
ax_c.set_xlim(-LIM_MIN, LIM_MIN)
ax_c.set_ylim(-LIM_MIN, LIM_MIN)
ax_c.set_aspect("equal", adjustable="box")
ax_c.axhline(0, color="gray", lw=0.8)
ax_c.axvline(0, color="gray", lw=0.8)
ax_c.set_title("Cercle trigonométrique")
ax_c.grid(alpha=0.3)
(circle,) = ax_c.plot([], [], color="black")
(radius,) = ax_c.plot([], [], color=COL_TH, lw=2, zorder=4)
(pt_c,) = ax_c.plot([], [], "o", color=COL_TH, ms=9, zorder=5)
(centre,) = ax_c.plot([], [], "k+", ms=10)

# Courbe
ax_g.set_xlim(X_MIN, X_MAX)
ax_g.axhline(0, color="gray", lw=0.8)
ax_g.axvline(0, color="gray", lw=0.8)
ks = np.arange(-2, 5)
ax_g.set_xticks(ks * np.pi)
ax_g.set_xticklabels(
    [r"$0$" if k == 0 else
     (r"$\pi$" if k == 1 else (r"$-\pi$" if k == -1 else rf"${k}\pi$"))
     for k in ks]
)
ax_g.set_xlabel(r"$x$")
ax_g.grid(alpha=0.3)
(curve,) = ax_g.plot([], [], color="tab:blue", lw=2)
(mid_line,) = ax_g.plot([], [], "--", color=COL_B, lw=1.2,
                        label="axe médian y = B")
(pt_g,) = ax_g.plot([], [], "o", color=COL_TH, ms=9, zorder=5)
ax_g.legend(loc="upper right")

# Trait reliant le point du cercle au point de la courbe
link = ConnectionPatch((0, 0), (0, 0), "data", "data",
                       axesA=ax_c, axesB=ax_g,
                       color=COL_TH, ls="--", lw=1)
fig.add_artist(link)


# ------------------------------- angles sur le cercle : x, ω·x, +φ, θ
class AngleArc:
    """Arc de cercle orienté (avec pointe de flèche) + étiquette."""

    def __init__(self, color, txt_color=None, lw=2.5):
        (self.line,) = ax_c.plot([], [], color=color, lw=lw, zorder=3)
        self.arrow = ax_c.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                            shrinkA=0, shrinkB=0, mutation_scale=14),
            zorder=3,
        )
        self.txt = ax_c.text(
            0, 0, "", color=txt_color or color, fontsize=11, zorder=6,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.8),
        )
        self.artists = (self.line, self.arrow, self.txt)

    def hide(self):
        for a in self.artists:
            a.set_visible(False)

    def set(self, cx, cy, r, a0, a1, label, r_txt=None):
        span = a1 - a0
        if abs(span) < 0.03:
            self.hide()
            return
        n = max(int(abs(span) / 0.05) + 2, 4)
        ang = np.linspace(a0, a1, n)
        self.line.set_data(cx + r * np.cos(ang), cy + r * np.sin(ang))
        # pointe de flèche à l'extrémité de l'arc
        tail = a1 - np.sign(span) * min(0.15, abs(span) / 2)
        self.arrow.xy = (cx + r * np.cos(a1), cy + r * np.sin(a1))
        self.arrow.xyann = (cx + r * np.cos(tail), cy + r * np.sin(tail))
        # étiquette près de la pointe de l'arc, un peu à l'extérieur
        am = a1 - np.sign(span) * min(0.35, abs(span) / 2)
        rt = r_txt if r_txt is not None else r + 0.32
        self.txt.set_position((cx + rt * np.cos(am), cy + rt * np.sin(am)))
        self.txt.set_text(label)
        for a in self.artists:
            a.set_visible(True)


arc_w = AngleArc(COL_W)                        # x -> ω·x       (bleu)
arc_phi = AngleArc(COL_PHI)                    # ω·x -> θ       (vert)

# x (gris) : rayon + point sur le cercle, comme pour θ (mauve)
(ray_x,) = ax_c.plot([], [], color=COL_X, lw=2, zorder=3)
(dot_x,) = ax_c.plot([], [], "o", color=COL_X, ms=9, zorder=4)
# ω·x (bleu) : rayon en pointillés + point plus petit
(ray_wx,) = ax_c.plot([], [], "--", color=COL_W, lw=1.5, zorder=3)
(dot_wx,) = ax_c.plot([], [], "o", color=COL_W, ms=5, zorder=4)


def point_label(color):
    return ax_c.text(
        0, 0, "", color=color, fontsize=11, zorder=6, ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                  alpha=0.8))


txt_x = point_label(COL_X)
txt_wx = point_label(COL_W)
txt_theta = point_label(COL_TH)

# Légende : cadre + une ligne colorée par angle, AU-DESSUS du cercle
LEG_X, LEG_Y, LEG_W = 0.01, 0.845, 0.28     # coordonnées de la figure
LEG_ROW = 0.030                              # hauteur d'une ligne
leg_box = Rectangle((LEG_X, LEG_Y), LEG_W, 0.13, transform=fig.transFigure,
                    fc="white", ec="0.7", alpha=0.9, zorder=6)
fig.add_artist(leg_box)
LEG_COLORS = (COL_X, COL_W, COL_PHI, COL_TH)
leg_txt = [
    fig.text(LEG_X + 0.012, LEG_Y + 0.02, "", fontsize=10.5,
             color=c, ha="left", va="center", zorder=7)
    for c in LEG_COLORS
]

CIRCLE_ARTISTS = (list(arc_w.artists) + list(arc_phi.artists)
                  + [ray_x, dot_x, txt_x, ray_wx, dot_wx, txt_wx, txt_theta,
                     leg_box] + leg_txt)


def update_circle_angles(A, w, phi, B, x0):
    """x (gris), ω·x (bleu, si utile), et les décalages x -> ωx -> θ."""
    cx, cy = 0.0, B
    rb = max(A, 1.0)                      # rayon de référence des arcs
    wx = w * x0
    theta = wx + phi
    show_w = abs(w - 1.0) > 0.005         # ω = 1 : ω·x = x, rien à montrer
    phi_zero = abs(phi) < 0.005           # φ = 0 : θ = ω·x
    coincide_x = (not show_w) and phi_zero  # θ = ω·x = x

    def pt(angle, r):
        return cx + r * np.cos(angle), cy + r * np.sin(angle)

    # éléments toujours visibles (réactivés si la case est recochée)
    show([ray_x, txt_theta, leg_box], True)

    # 1) x (gris) : rayon et point d'intersection avec le cercle
    #    (masqués si θ = x, pour ne pas se superposer à θ)
    px, py = pt(x0, A)
    ray_x.set_data([cx, px], [cy, py])
    dot_x.set_data([px], [py])
    txt_x.set_position(pt(x0, A + 0.35))
    txt_x.set_text(r"$x$")
    show([dot_x, txt_x], not coincide_x)

    # 2) pulsation : x -> ω·x (bleu), flèche + rayon + petit point
    if show_w:
        arc_w.set(cx, cy, 0.55 * rb, x0, wx, r"$\times\omega$")
        if phi_zero:                      # θ = ω·x : on ne double pas θ
            show([ray_wx, dot_wx, txt_wx], False)
        else:
            qx, qy = pt(wx, A)
            ray_wx.set_data([cx, qx], [cy, qy])
            dot_wx.set_data([qx], [qy])
            txt_wx.set_position(pt(wx, A + 0.35))
            txt_wx.set_text(r"$\omega x$")
            show([ray_wx, dot_wx, txt_wx], True)
    else:
        show([ray_wx, dot_wx, txt_wx], False)
        arc_w.hide()

    # 3) déphasage : ω·x -> θ = ω·x + φ (vert)
    arc_phi.set(cx, cy, 0.70 * rb, wx, theta, r"$+\varphi$")

    # 4) θ (mauve, au bout du rayon mauve)
    txt_theta.set_position(pt(theta, A + 0.35))
    txt_theta.set_text(r"$\theta$")

    # 5) légende : une ligne par angle affiché
    rows = [(leg_txt[0], rf"$x = {x0:.2f}$ rad")]
    if show_w and not phi_zero:
        rows.append((leg_txt[1],
                     rf"$\omega x = {w:.1f}\times{x0:.2f} = {wx:.2f}$ rad"))
    rows.append((leg_txt[2], rf"$\varphi = {phi:+.2f}$ rad  (phase à l'origine)"))
    rows.append((leg_txt[3],
                 rf"$\theta = \omega x + \varphi = {theta:.2f}$ rad"))
    h = LEG_ROW * len(rows) + 0.012
    leg_box.set_height(h)
    show(leg_txt, False)
    for i, (tx, label) in enumerate(rows):
        tx.set_text(label)
        tx.set_position((LEG_X + 0.012,
                         LEG_Y + h - 0.021 - LEG_ROW * i))
        tx.set_visible(True)


# ------------------------------------- annotations des paramètres (courbe)
def make_arrow(color, style="<->"):
    a = ax_g.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle=style, color=color, lw=2,
                        shrinkA=0, shrinkB=0, mutation_scale=14),
        zorder=4,
    )
    a.set_visible(False)
    return a


def make_label(color, **kw):
    txt = ax_g.text(
        0, 0, "", color=color, fontsize=12, zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                  alpha=0.75),
        **kw,
    )
    txt.set_visible(False)
    return txt


# B : flèche de y = 0 à y = B, près du bord gauche
arr_B = make_arrow(COL_B)
txt_B = make_label(COL_B, ha="left", va="center")

# A : flèche de l'axe médian jusqu'au maximum
arr_A = make_arrow(COL_A)
txt_A = make_label(COL_A_TXT, ha="left", va="center")

# T = 2π/ω : flèche entre deux maxima consécutifs
arr_T = make_arrow(COL_W)
txt_T = make_label(COL_W, ha="center", va="top")

# déphasage -φ/ω : décalage horizontal (de x = 0 à x = -φ/ω), point, valeur
arr_phi = make_arrow(COL_PHI, style="->")
(pt_phi,) = ax_g.plot([], [], "o", color=COL_PHI, mec="k", ms=10, zorder=6)
pt_phi.set_visible(False)
txt_phi = make_label(COL_PHI, ha="left", va="top")


# ----------------------------------------------------------------- curseurs
def make_slider(y, label, vmin, vmax, vinit, step=None, color=None,
                valfmt=None):
    ax = fig.add_axes([0.22, y, 0.45, 0.03])
    kw = {}
    if valfmt is not None:
        kw["valfmt"] = valfmt
    if color is not None:
        kw["color"] = color
        kw["handle_style"] = {"facecolor": color, "edgecolor": "k",
                              "size": 12}
    s = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step, **kw)
    if color is not None:
        s.label.set_color(color)
    return s


s_A = make_slider(0.25, r"$A$ (amplitude)", 0.0, 4.0, A0, 0.1, COL_A)
s_w = make_slider(0.20, r"$\omega$ (pulsation)", 0.1, 4.0, W0, 0.1, COL_W)
# bornes ±3.15 (≈ ±π) : multiples de 0,05, donc φ = 0 est atteignable
s_phi = make_slider(0.15, r"$\varphi$ (phase à l'origine)", -3.15, 3.15, PHI0,
                    0.05, COL_PHI, valfmt="%.2f")
s_B = make_slider(0.10, r"$B$ (décalage)", -2.0, 2.0, B0, 0.1, COL_B)
s_x = make_slider(0.05, r"$x$ (position)", 0.0, X_MAX, X0, color=COL_X)

ax_reset = fig.add_axes([0.74, 0.05, 0.10, 0.05])
btn_reset = Button(ax_reset, "Réinitialiser")
ax_play = fig.add_axes([0.85, 0.05, 0.10, 0.05])
btn_play = Button(ax_play, "▶ Lecture")

# Cases à cocher : quelles annotations afficher
ANN_GROUPS = {
    "A": (COL_A_TXT, lambda: (arr_A, txt_A)),
    "ω": (COL_W, lambda: (arr_T, txt_T)),
    "φ": (COL_PHI, lambda: (arr_phi, pt_phi, txt_phi)),
    "B": (COL_B, lambda: (arr_B, txt_B)),
    "x": (COL_TH, lambda: CIRCLE_ARTISTS),
}
ANN_LABELS = {
    "A": r"$A$  (amplitude)",
    "ω": r"$\omega$  (période $T$)",
    "φ": r"$-\varphi/\omega$  (déphasage)",
    "B": r"$B$  (décalage)",
    "x": r"angles $x$, $\omega x$, $\varphi$, $\theta$ (cercle)",
}
show_ann = {k: True for k in ANN_GROUPS}

ax_chk = fig.add_axes([0.74, 0.11, 0.23, 0.19])
_names = list(ANN_GROUPS)
_cols = [ANN_GROUPS[k][0] for k in _names]
try:
    chk = CheckButtons(ax_chk, [ANN_LABELS[k] for k in _names],
                       [True] * len(_names),
                       check_props={"color": _cols, "s": 90},
                       frame_props={"edgecolor": _cols, "s": 160,
                                    "facecolor": "none"})
except TypeError:                       # matplotlib plus ancien
    chk = CheckButtons(ax_chk, [ANN_LABELS[k] for k in _names],
                       [True] * len(_names))
for lab, c in zip(chk.labels, _cols):
    lab.set_color(c)
    lab.set_fontsize(11)


def on_check(label):
    key = _names[[ANN_LABELS[k] for k in _names].index(label)]
    show_ann[key] = not show_ann[key]
    update()


chk.on_clicked(on_check)

# ------------------------------------------------- titre : g(x) = ... coloré
FS = 16                 # taille de la formule
FS_LET = 14             # taille des lettres au-dessus des valeurs
Y_FORM = 0.935          # hauteur de la formule (coord. figure)
Y_LET = 0.963           # base des lettres
Y_UNDER = Y_FORM - 0.036  # trait mauve sous la parenthèse
Y_THETA = Y_UNDER - 0.006  # sommet de l'étiquette θ sous le trait


def boxed_text(color):
    """Texte encadré : fond translucide + contour de la couleur."""
    return fig.text(
        0, Y_FORM, "", fontsize=FS, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc=to_rgba(color, 0.25),
                  ec=color, lw=2),
    )


def plain_text():
    return fig.text(0, Y_FORM, "", fontsize=FS, ha="left", va="center")


def letter_text(tex, color):
    return fig.text(0, Y_LET, tex, fontsize=FS_LET, color=color,
                    ha="center", va="bottom")


t_g = plain_text()
t_A, l_A = boxed_text(COL_A), letter_text(r"$A$", COL_A)
t_sin = plain_text()
t_lp = plain_text()                       # parenthèse ouvrante
t_w, l_w = boxed_text(COL_W), letter_text(r"$\omega$", COL_W)
t_x = plain_text()
t_sg1 = plain_text()
t_phi, l_phi = boxed_text(COL_PHI), letter_text(r"$\varphi$", COL_PHI)
t_rp = plain_text()                       # parenthèse fermante
t_sg2 = plain_text()
t_B, l_B = boxed_text(COL_B), letter_text(r"$B$", COL_B)

# Soulignement mauve de la parenthèse + étiquette θ en dessous
under = Line2D([0, 0], [Y_UNDER, Y_UNDER], color=COL_TH, lw=2.5,
               transform=fig.transFigure, solid_capstyle="round")
fig.add_artist(under)
lab_theta = fig.text(0, Y_THETA, r"$\theta$", fontsize=FS_LET, color=COL_TH,
                     ha="center", va="top")

# (artiste, lettre au-dessus ou None, encadré ?, espace après en unités de gap)
ITEMS = [
    (t_g, None, False, 1),
    (t_A, l_A, True, 1),
    (t_sin, None, False, 0),
    (t_lp, None, False, 0.4),
    (t_w, l_w, True, 1),
    (t_x, None, False, 1),
    (t_sg1, None, False, 1),
    (t_phi, l_phi, True, 1),
    (t_rp, None, False, 1),
    (t_sg2, None, False, 1),
    (t_B, l_B, True, 0),
]


def get_renderer():
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backend_bases import _get_renderer
        return _get_renderer(fig)


def layout_title():
    """Place les morceaux côte à côte, centrés sur la figure."""
    r = get_renderer()
    W = fig.bbox.width
    pad_px = 0.3 * FS * fig.dpi / 72         # marge interne du cadre
    gap = 0.30 * FS * fig.dpi / 72           # espace entre deux morceaux

    widths = [a.get_window_extent(r).width + (2 * pad_px if boxed else 0)
              for a, _, boxed, _ in ITEMS]
    total = sum(w + gap * g for w, (_, _, _, g) in zip(widths, ITEMS))
    xpx = 0.5 * W - total / 2

    start = {}
    for (art, let, boxed, g), w in zip(ITEMS, widths):
        art.set_position(((xpx + (pad_px if boxed else 0)) / W, Y_FORM))
        if let is not None:
            let.set_position(((xpx + w / 2) / W, Y_LET))
        start[art] = (xpx, w)
        xpx += w + gap * g

    # soulignement : de l'avant de « ( » à l'arrière de « ) »
    x0 = start[t_lp][0]
    x1 = start[t_rp][0] + start[t_rp][1]
    under.set_data([x0 / W, x1 / W], [Y_UNDER, Y_UNDER])
    lab_theta.set_position((0.5 * (x0 + x1) / W, Y_THETA))


# --------------------------- sous-titre : x -> ωx -> θ (symboles colorés)
FS_SUB = 12
Y_SUB = 0.815                         # juste au-dessus de la courbe
SUB_N = 8
SUB_COLORS = (COL_X, "k", "k", COL_W, "k", "k", COL_TH, "k")
sub_txt = [fig.text(0, Y_SUB, "", fontsize=FS_SUB, color=c, ha="left",
                    va="baseline") for c in SUB_COLORS]


def update_subtitle(x0, w, theta):
    vals = [r"$x$", rf"$= {x0:.2f}$", r"$\to$",
            r"$\omega x$", rf"$= {w * x0:.2f}$", r"$\to$",
            r"$\theta = \omega x + \varphi$",
            rf"$= {theta:.2f}\ \mathrm{{rad}}$"]
    for tx, v in zip(sub_txt, vals):
        tx.set_text(v)
    r = get_renderer()
    W = fig.bbox.width
    gap = 0.25 * FS_SUB * fig.dpi / 72
    widths = [tx.get_window_extent(r).width for tx in sub_txt]
    total = sum(widths) + gap * (SUB_N - 1)
    pos = ax_g.get_position()
    xpx = (pos.x0 + pos.width / 2) * W - total / 2
    for tx, wd in zip(sub_txt, widths):
        tx.set_position((xpx / W, Y_SUB))
        xpx += wd + gap


def sign(v):
    return r"$-$" if round(v, 2) < 0 else r"$+$"


def show(artists, flag):
    for a in artists:
        a.set_visible(flag)


# --------------------------------------------------------------- mise à jour
def update_limits(A, B):
    """Limite y = ±(⌊A + |B|⌋ + 1), au minimum ±2 : paliers à l'unité."""
    lim = max(LIM_MIN, int(np.floor(round(A + abs(B), 6))) + 1)
    if lim != lim_state["v"]:
        lim_state["v"] = lim
        ax_c.set_xlim(-lim, lim)
        ax_c.set_ylim(-lim, lim)          # ax_g partage l'axe y


def compute_annotations(A, w, phi, B):
    """Calcule les annotations A, T, φ et B sur le graphe de g."""
    T = 2 * np.pi / w
    x_ph = -phi / w                       # montée par l'axe médian
    has_amp = A >= 0.05

    # ---- B : décalage vertical (de y = 0 à y = B)
    if abs(B) >= 0.05:
        xB = X_MIN + 0.35
        arr_B.xy = (xB, B)
        arr_B.xyann = (xB, 0)
        txt_B.set_position((xB + 0.15, B / 2))
        txt_B.set_text(rf"$B={B:.1f}$")
        show((arr_B, txt_B), True)
    else:
        show((arr_B, txt_B), False)

    if not has_amp:                       # courbe plate : rien d'autre à voir
        show((arr_A, txt_A, arr_T, txt_T, arr_phi, pt_phi, txt_phi), False)
        return

    # ---- φ : point d'intersection avec l'axe médian, d'abscisse -φ/ω
    in_view = X_MIN <= x_ph <= X_MAX
    # si -φ/ω sort de la fenêtre, on montre la montée équivalente (à kT près)
    x_vis = x_ph if in_view else X_MIN + (x_ph - X_MIN) % T
    if x_vis <= X_MAX:
        pt_phi.set_data([x_vis], [B])
        label = rf"$-\varphi/\omega={x_ph:.2f}$"
        if not in_view:
            label += "\n(à $kT$ près)"
        txt_phi.set_text(label)
        txt_phi.set_position((x_vis + 0.12, B - 0.15))
        show((pt_phi, txt_phi), True)
    else:
        show((pt_phi, txt_phi), False)

    # flèche du décalage horizontal : de x = 0 jusqu'à x = -φ/ω
    if in_view and abs(x_ph) > 0.05:
        arr_phi.xy = (x_ph, B)
        arr_phi.xyann = (0, B)
        arr_phi.set_visible(True)
    else:
        arr_phi.set_visible(False)

    # ---- A et T : on s'appuie sur un maximum visible
    lim = lim_state["v"]
    lo = X_MIN + 0.8                      # on évite le bord (où est B)
    x_pk = lo + (x_ph + T / 4 - lo) % T   # abscisse d'un maximum
    top = B + A
    if x_pk <= X_MAX and top <= lim:
        # amplitude : de l'axe médian au maximum
        arr_A.xy = (x_pk, top)
        arr_A.xyann = (x_pk, B)
        txt_A.set_text(rf"$A={A:.1f}$")
        txt_A.set_position((x_pk + 0.12, B + A / 2))
        show((arr_A, txt_A), True)
        # période : entre deux maxima consécutifs
        if x_pk + T <= X_MAX:
            arr_T.xy = (x_pk + T, top)
            arr_T.xyann = (x_pk, top)
            txt_T.set_text(rf"$T=2\pi/\omega={T:.2f}$")
            txt_T.set_position((x_pk + T / 2, top - 0.12))
            show((arr_T, txt_T), True)
        else:
            show((arr_T, txt_T), False)
    else:
        show((arr_A, txt_A, arr_T, txt_T), False)


def update_annotations(A, w, phi, B, x0):
    compute_annotations(A, w, phi, B)
    update_circle_angles(A, w, phi, B, x0)
    for key, (_, getter) in ANN_GROUPS.items():
        if not show_ann[key]:
            show(getter(), False)       # case décochée : on masque


def update(_=None):
    A, w, phi, B, x0 = s_A.val, s_w.val, s_phi.val, s_B.val, s_x.val
    theta = w * x0 + phi
    px, py = A * np.cos(theta), B + A * np.sin(theta)   # point sur le cercle

    update_limits(A, B)

    circle.set_data(A * np.cos(t), B + A * np.sin(t))
    centre.set_data([0], [B])
    radius.set_data([0, px], [B, py])
    pt_c.set_data([px], [py])

    curve.set_data(x, A * np.sin(w * x + phi) + B)
    mid_line.set_data([X_MIN, X_MAX], [B, B])
    pt_g.set_data([x0], [py])

    link.xy1 = (px, py)
    link.xy2 = (x0, py)

    update_annotations(A, w, phi, B, x0)

    # Titre : formule colorée (les signes sont hors des cadres)
    t_g.set_text(r"$g(x)=$")
    t_A.set_text(rf"${A:.1f}$")
    t_sin.set_text(r"$\sin$")
    t_lp.set_text(r"$($")
    t_w.set_text(rf"${w:.1f}$")
    t_x.set_text(r"$x$")
    t_sg1.set_text(sign(phi))
    t_phi.set_text(rf"${abs(phi):.2f}$")
    t_rp.set_text(r"$)$")
    t_sg2.set_text(sign(B))
    t_B.set_text(rf"${abs(B):.1f}$")
    layout_title()

    update_subtitle(x0, w, theta)
    fig.canvas.draw_idle()


for s in (s_A, s_w, s_phi, s_B, s_x):
    s.on_changed(update)
fig.canvas.mpl_connect("resize_event", lambda e: update())

# ----------------------------------------------------------------- animation
timer = fig.canvas.new_timer(interval=30)
state = {"playing": False}


def tick():
    s_x.set_val((s_x.val + 0.05) % X_MAX)


timer.add_callback(tick)


def toggle(_):
    state["playing"] = not state["playing"]
    if state["playing"]:
        timer.start()
        btn_play.label.set_text("⏸ Pause")
    else:
        timer.stop()
        btn_play.label.set_text("▶ Lecture")
    fig.canvas.draw_idle()


def reset(_):
    for s in (s_A, s_w, s_phi, s_B, s_x):
        s.reset()


btn_play.on_clicked(toggle)
btn_reset.on_clicked(reset)

update()
plt.show()