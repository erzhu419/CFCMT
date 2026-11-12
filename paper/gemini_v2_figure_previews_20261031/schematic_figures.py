"""Python reconstructions of the two supplied conceptual figure definitions."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as DrawingPath


mpl.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "figure.dpi": 300,
}
STATUS = "CONCEPTUAL PREVIEW — proposed framework, not frozen V93 implementation"
BLUE, ORANGE, TEAL = "#286587", "#AD6C2B", "#237A6D"


def _box(ax, xy, width, height, *, fill, edge, dashed=False, radius=0.08):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=0.9,
        linestyle=(0, (4, 2.5)) if dashed else "-", zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color="#34495E", dashed=False, curve=0.0):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7.0, linewidth=0.85,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}", shrinkA=1.0, shrinkB=2.0,
        zorder=3,
    )
    ax.add_patch(arrow)


def _route(ax, points, *, color="#34495E", dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linewidth=0.85,
            linestyle="--" if dashed else "-", zorder=1)
    _arrow(ax, points[-2], points[-1], color=color, dashed=dashed)


def _label(ax, x, y, text, *, color="#263746", size=6.0):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4), zorder=4)


def _status(fig):
    fig.text(0.5, 0.018, STATUS, ha="center", va="bottom",
             fontsize=6.2, fontweight="bold", color="#9A4D18")


def draw_timeline():
    """Return the movement-level phase transition and rollout-window diagram."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 2.55))
        fig.subplots_adjust(left=0.025, right=0.97, bottom=0.14, top=0.96)
        ax.set(xlim=(-0.05, 9.6), ylim=(-1.2, 2.25))
        ax.axis("off")

        _arrow(ax, (0.0, 0.0), (9.25, 0.0), color="#35404A")
        ax.text(9.25, 0.16, "Control clock $t$", ha="right", va="bottom",
                fontsize=7.0, fontweight="bold")

        _box(ax, (0.4, 0.35), 3.2, 1.0, fill="#E8F1F5", edge=BLUE)
        ax.text(2.0, 1.1, r"Active phase $\phi_{i,t}$", ha="center",
                va="center", fontsize=7.5, fontweight="bold", color=BLUE)
        _box(ax, (0.55, 0.45), 2.9, 0.4, fill="#FFF0DE", edge=ORANGE,
             radius=0.035)
        ax.text(2.0, 0.65, r"Elapsed green $\tau_{i,t}\geq\tau_{\min}$",
                ha="center", va="center", fontsize=7.2)

        _box(ax, (3.65, 0.35), 1.8, 1.0, fill="#FCEAEA", edge="#AC4545",
             dashed=True)
        ax.text(4.55, 1.1, r"If switch: $a\ne\phi_{i,t}$", ha="center",
                va="center", fontsize=6.8, fontweight="bold", color="#A23939")
        ax.text(4.55, 0.70, r"Clearance $\tau_{\mathrm{yel}}$"+"\n"+
                r"Lost cost $\hat c_{i,t}(a)$", ha="center", va="center",
                fontsize=6.8, linespacing=1.35)

        _box(ax, (5.5, 0.35), 3.05, 1.0, fill="#E8F8F5", edge=TEAL)
        ax.text(7.02, 1.1, r"Candidate phase $a\in\mathcal{A}_{\mathrm{safe}}$",
                ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=TEAL)
        _box(ax, (5.65, 0.45), 2.75, 0.4, fill="#CFECE6", edge=TEAL,
             radius=0.035)
        ax.text(7.02, 0.65,
                r"Discharge $\hat d_{m,t}$ & inflow $\hat v_{m,t+1}$",
                ha="center", va="center", fontsize=7.0)

        ticks = [
            (0.4, r"$t-\tau_{i,t}$", "#35404A", 0.1),
            (3.6, "$t$\n(Decision)", "#A23939", 0.35),
            (5.48, r"$t+\tau_{\mathrm{yel}}$", "#35404A", 0.1),
            (8.5, "$t+\Delta t$\n(Evaluation)", BLUE, 0.35),
        ]
        for x, text, color, top in ticks:
            ax.plot([x, x], [top, -0.12], color=color, linewidth=0.9)
            ax.text(x, -0.2, text, ha="center", va="top", fontsize=6.8,
                    color=color, linespacing=1.15)

        # A downward brace marks the same counterfactual interval as the draft.
        x0, x1, y = 3.6, 8.5, -0.72
        xm = (x0+x1)/2
        vertices = [(x0,y), (x0,y-0.12), (xm-0.22,y-0.12), (xm,y-0.23),
                    (xm+0.22,y-0.12), (x1,y-0.12), (x1,y)]
        codes = [DrawingPath.MOVETO, DrawingPath.CURVE4, DrawingPath.CURVE4,
                 DrawingPath.CURVE4, DrawingPath.CURVE4, DrawingPath.CURVE4,
                 DrawingPath.CURVE4]
        ax.add_patch(PathPatch(DrawingPath(vertices,codes), fill=False,
                               edgecolor="#4B5055", linewidth=0.85))
        ax.text(xm, -1.03,
                r"Counterfactual rollout $\Delta\widehat W(s_{i,t},a;\Theta_k)$ across $\mathcal{M}_i$",
                ha="center", va="center", fontsize=6.8)

        _arrow(ax, (3.6,1.43), (8.45,1.43), color="#704691",
               dashed=True, curve=-0.18)
        _label(ax, 6.05, 2.0,
               r"Right-of-way shield $\Pi_{\mathrm{safe}}:\mathcal{A}_{\mathrm{safe}}(s_{i,t})$",
               color="#704691", size=7.0)
        _status(fig)
        return fig


def _node(ax, x, y, width, height, title, lines, edge, *, title_size=6.8,
          body_size=6.3, fill="white"):
    _box(ax, (x-width/2,y-height/2), width, height, fill=fill, edge=edge)
    ax.text(x,y+height/2-0.20,title,ha="center",va="top",fontsize=title_size,
            fontweight="bold",color="#243540",linespacing=1.15,zorder=4)
    body_top=y+height/2-0.64
    body_bottom=y-height/2+0.26
    if len(lines)==1:
        ys=[(body_top+body_bottom)/2]
    else:
        ys=[body_top+(body_bottom-body_top)*i/(len(lines)-1)
            for i in range(len(lines))]
    for text,yy in zip(lines,ys):
        ax.text(x,yy,text,ha="center",va="center",fontsize=body_size,
                color="#263746",zorder=4)


def draw_framework():
    """Return the three-panel source/mechanism/gate conceptual framework."""
    with mpl.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.16,4.65))
        fig.subplots_adjust(left=0.015,right=0.985,bottom=0.105,top=0.985)
        ax.set(xlim=(-0.18,18.18),ylim=(-0.65,7.95))
        ax.axis("off")
        panels=[
            (0,5.6,"#E8F1F5",BLUE,"I. Multi-city source bank\nand shielding"),
            (5.9,5.8,"#FDF2E9",ORANGE,"II. Causal mechanism\ndecomposition"),
            (12.0,6.0,"#E8F8F5",TEAL,"III. Epistemic gate\nand action arbitration"),
        ]
        for x,width,fill,edge,title in panels:
            _box(ax,(x,0.15),width,7.6,fill=fill,edge=edge,dashed=True,radius=0.12)
            ax.text(x+width/2,7.57,title,ha="center",va="top",fontsize=8.0,
                    fontweight="bold",color=edge,linespacing=1.15,zorder=4)

        _node(ax,2.8,5.90,5.0,1.75,
              r"Multi-city source bank ($K=10$)",
              ["German urban + synthetic grids + Salt Lake",
               r"Bootstrap ensembles $\{\mathcal{M}_k\}_{k=1}^{K}$"],BLUE,
              body_size=5.9)
        _node(ax,2.8,3.6,4.8,1.5,"Right-of-way phase shielding",
              [r"$\Pi_{\mathrm{safe}}:(\tau_{\min},\tau_{\max},\tau_{\mathrm{yel}})$",
               r"$\longrightarrow\mathcal{A}_{\mathrm{safe}}(s_{i,t})$"],ORANGE,
              fill="#FFF7DC")
        _node(ax,2.8,1.3,5.0,1.75,"Unseen target road network",
              [r"$\mathcal{D}_{\mathrm{tgt}}$: Los Angeles and Jinan",
               r"Incidence matrix $B_i$ + local state $s_{i,t}$"],BLUE)

        _node(ax,8.8,5.9,5.1,1.8,"1) Inflow and 2) service discharge",
              [r"$\hat v_{m,t+1}=f_{\mathrm{arr}}(v_{m,t},q_{m,t};\theta_{\mathrm{arr}})$",
               r"$\hat d_{m,t}=u_{m,t}(1-\mathbb{I}_{\mathrm{sw}}\kappa_{\mathrm{lost}})$",
               r"$\times f_{\mathrm{srv}}(q_{m,t},d_{\mathrm{down}})$"],ORANGE,
              body_size=6.0)
        _node(ax,8.8,3.6,5.1,1.95,"3) Queue waiting and 4) switch cost",
              [r"$\hat w_{m,t+1}=f_{\mathrm{wait}}(w_{m,t},q_{m,t},$",
               r"$\hat v_{m,t+1},\hat d_{m,t})$",
               r"$\hat c_{i,t}(a)=\mathbb{I}(a\ne\phi_{i,t})$",
               r"$\times f_{\mathrm{sw}}(q_{\phi_{i,t}},\tau_{\mathrm{yel}})$"],ORANGE,
              body_size=6.1)
        _node(ax,8.8,1.3,5.1,1.55,r"Nominal target backbone ($\alpha=0$)",
              ["Local counterfactual advantage",
               r"$\widehat\Delta_{\mathrm{tgt}}(s_{i,t},a)$"],ORANGE)

        _node(ax,15.0,5.9,5.3,1.8,r"Epistemic compatibility gate ($\beta\geq0$)",
              [r"Source mean $\mu_k(s_{i,t},a)$ and dispersion $\sigma_k(s_{i,t},a)$",
               r"$\widetilde\Delta_{\mathrm{src}}^{(k)}=\mu_k(s_{i,t},a)$",
               r"$-\beta\sigma_k(s_{i,t},a)$"],TEAL,body_size=5.9)
        _node(ax,15.0,3.6,5.3,1.95,r"Structural blending ($\alpha\in[0,1]$)",
              [r"$\widehat{\mathcal{J}}_{\mathrm{CFCMT}}(s_{i,t},a)=$",
               r"$(1-\alpha)\widehat\Delta_{\mathrm{tgt}}(s_{i,t},a)$",
               r"$+\alpha\sum_{k\in\mathcal{S}^\star}w_k(s_{i,t})\,$",
               r"$\times\widetilde\Delta_{\mathrm{src}}^{(k)}(s_{i,t},a)$"],TEAL,
              body_size=6.1)
        _node(ax,15.0,1.25,5.3,1.85,"Zero-shot shielded action execution",
              [r"$a_{i,t}^\star=\arg\max_{a\in\mathcal{A}_{\mathrm{safe}}(s_{i,t})}$",
               r"$\widehat{\mathcal{J}}_{\mathrm{CFCMT}}(s_{i,t},a;\alpha,\beta)$",
               r"Fallback to $\widehat\Delta_{\mathrm{tgt}}$ as $\sigma_k\to\infty$"],TEAL,
              body_size=6.0)

        _arrow(ax,(5.3,5.9),(6.25,5.9))
        _arrow(ax,(5.2,3.6),(6.25,3.6))
        _arrow(ax,(5.3,1.3),(6.25,1.3))
        _arrow(ax,(2.8,2.175),(2.8,2.85),color=BLUE)
        _label(ax,2.8,2.53,r"Observe $s_{i,t},\tau_{i,t}$",color=BLUE,size=5.8)
        _arrow(ax,(8.8,5.0),(8.8,4.575),color=ORANGE,dashed=True)
        _arrow(ax,(11.35,5.9),(12.35,5.9))
        _route(ax,[(11.35,3.6),(11.85,3.6),(11.85,5.3),(12.35,5.3)])
        _arrow(ax,(15.0,5.0),(15.0,4.575),color="#AA4444")
        _label(ax,16.65,4.79,r"Pessimistic $\widetilde\Delta_{\mathrm{src}}^{(k)}$",
               color="#AA4444",size=5.3)
        _route(ax,[(11.35,1.3),(11.65,1.3),(11.65,3.05),(12.35,3.05)])
        _label(ax,9.15,2.34,r"Anchor $(1-\alpha)\widehat\Delta_{\mathrm{tgt}}$",size=5.6)
        _arrow(ax,(15.0,2.625),(15.0,2.175),color=TEAL)
        _label(ax,16.75,2.4,r"Rank $\forall a\in\mathcal{A}_{\mathrm{safe}}$",color=TEAL,size=5.5)
        _route(ax,[(15.0,0.325),(15.0,-0.32),(2.8,-0.32),(2.8,0.425)],color=BLUE)
        _label(ax,8.8,-0.31,r"Execute phase $a_{i,t}^\star$ in the target SUMO corridor",color=BLUE,size=6.0)
        _status(fig)
        return fig


if __name__=="__main__":
    destination=Path(__file__).resolve().parent/"figures"
    destination.mkdir(exist_ok=True)
    for stem,draw in [("fig1_mechanism_timeline",draw_timeline),
                      ("fig2_framework_overview",draw_framework)]:
        figure=draw()
        for extension in ("png","pdf","svg"):
            figure.savefig(destination/f"{stem}.{extension}",dpi=300,
                           bbox_inches="tight",pad_inches=0.06)
        plt.close(figure)
        print(f"Saved {stem}")
