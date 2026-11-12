"""Diagrams of the implemented V93 fitting and phase-control procedure."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
}
BLUE, ORANGE, TEAL = "#286587", "#AD6C2B", "#237A6D"


def _box(ax, x, y, width, height, *, fill="white", edge="#41505A",
         dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor=fill, edgecolor=edge, linewidth=0.85,
        linestyle=(0, (4, 2.5)) if dashed else "-", zorder=2,
    ))


def _arrow(ax, start, end, *, color="#41505A", dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=6.5, linewidth=0.8,
        color=color, linestyle="--" if dashed else "-", shrinkA=1,
        shrinkB=2, zorder=3,
    ))


def _route(ax, points, *, color="#41505A", dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linewidth=0.8,
            linestyle="--" if dashed else "-", zorder=1)
    _arrow(ax, points[-2], points[-1], color=color, dashed=dashed)


def _node(ax, x, y, width, height, title, lines, *, edge,
          fill="white", title_size=7.0, body_size=6.3):
    _box(ax, x-width/2, y-height/2, width, height, edge=edge, fill=fill)
    ax.text(x, y+height/2-height*0.075, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color="#243540",
            linespacing=1.2, zorder=4)
    title_rows = title.count("\n") + 1
    top = y+height/2-height*(0.37+0.15*(title_rows-1))
    bottom = y-height/2+height*0.14
    positions = [(top+bottom)/2] if len(lines)==1 else [
        top+(bottom-top)*i/(len(lines)-1) for i in range(len(lines))
    ]
    for line, yy in zip(lines, positions):
        ax.text(x, yy, line, ha="center", va="center", fontsize=body_size,
                color="#263746", zorder=4)


def draw_timeline():
    """Return the offline matched-label timeline and frozen runtime chain."""
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 3.5))
        fig.subplots_adjust(left=0.025, right=0.98, bottom=0.105, top=0.97)
        ax.set(xlim=(0, 100), ylim=(0, 45))
        ax.axis("off")

        ax.text(1.5, 44.3, "(a) Offline matched-action label collection",
                fontsize=8.2, fontweight="bold", va="top", color=BLUE)
        _node(ax, 10, 34.5, 17, 10.5, "Warm-up and\nsnapshot",
              ["60-s warm-up", "Simulator +", "executor state"], edge=BLUE,
              fill="#E8F1F5", title_size=6.6, body_size=6.0)
        _arrow(ax, (18.5,34.5), (21.0,34.5), color=BLUE)

        # Widths of the two branch segments are proportional to 10 s and 50 s.
        _node(ax, 27.15, 34.5, 12.3, 10.5, "Focal phase\nrequest",
              ["One action", "$a$ for 10 s"], edge=ORANGE,
              fill="#FDF2E9", title_size=6.6, body_size=6.1)
        _node(ax, 65.5, 34.5, 64, 10.5, "Common continuation",
              ["PhasePressure for the next five 10-s intervals",
               "Other signals use PhasePressure throughout"], edge=TEAL,
              fill="#E8F8F5", body_size=6.8)
        _arrow(ax, (21.0,27.0), (98.3,27.0))
        for x,label in [(21.0,"0 s"),(33.8,"10 s"),(97.5,"60 s")]:
            ax.plot([x,x],[26.2,27.8],color="#41505A",linewidth=0.75)
            ax.text(x,25.3,label,ha="center",va="top",fontsize=6.3)
        ax.text(60,22.4,
                "Restore the same snapshot before each alternative action",
                ha="center", va="center", fontsize=6.8, color=BLUE)
        ax.text(60,19.9,
                r"Label: $z_g(a)=[Y_g(a)-Y_g(r_g)]/\sigma_g$",
                ha="center", va="center", fontsize=7.0)
        ax.text(60,17.8,
                r"$Y_g$: mean active + pending vehicle load per controlled lane",
                ha="center", va="center", fontsize=6.3)

        ax.plot([1.5,98.5],[15.6,15.6],color="#B9C2C8",linewidth=0.6,
                linestyle=(0,(3,3)))
        ax.text(1.5,14.4,"(b) Frozen online control: one decision every 10 s",
                fontsize=8.2,fontweight="bold",va="top",color=TEAL)
        nodes=[
            (11.0,17.0,"Observe state",["Queues, occupancy,", "current phase and time"]),
            (34.2,21.0,"Frozen predictions",["Score feasible phases", "Select the minimum score"]),
            (60.0,24.5,"Guard + coordination",["Pressure-reference guard", "Sparse learned interventions"]),
            (86.9,22.0,"Common phase executor",["Minimum-green constraints", "Program yellow / all-red"]),
        ]
        for x,width,title,lines in nodes:
            _node(ax,x,7.0,width,8.2,title,lines,edge=TEAL,
                  fill="#F5FAF8",title_size=6.5,body_size=6.0)
        for start,end in [((19.5,7.0),(23.7,7.0)),
                          ((44.7,7.0),(47.75,7.0)),
                          ((72.25,7.0),(75.9,7.0))]:
            _arrow(ax,start,end,color=TEAL)
        fig.text(0.5,0.025,
                 "V93: 60-s offline labels; frozen 10-s decisions; shared phase executor",
                 ha="center",fontsize=6.5,color="#46535C")
        return fig


def draw_framework():
    """Return the data, fitting/selection, and frozen-deployment workflow."""
    with mpl.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.16,5.0))
        fig.subplots_adjust(left=0.015,right=0.985,bottom=0.09,top=0.98)
        ax.set(xlim=(-0.2,18.2),ylim=(-0.55,8.7))
        ax.axis("off")
        panels=[
            (0,5.6,"#E8F1F5",BLUE,"I. Offline data"),
            (5.9,5.8,"#FDF2E9",ORANGE,"II. Fitting and selection"),
            (12.0,6.0,"#E8F8F5",TEAL,"III. Frozen control"),
        ]
        for x,width,fill,edge,title in panels:
            _box(ax,x,0.12,width,8.35,fill=fill,edge=edge,dashed=True)
            ax.text(x+width/2,8.2,title,ha="center",va="top",fontsize=8.2,
                    fontweight="bold",color=edge)

        _node(ax,2.8,6.45,5.0,2.4,"Multi-city source bank",
              ["18 networks in 7 source groups",
               "Atlanta · Cologne · Hangzhou",
               "Ingolstadt · New York",
               "Synthetic · Salt Lake City"],edge=BLUE,body_size=6.1)
        _node(ax,2.8,3.75,5.0,2.25,"Target adaptation data",
              ["Seeds 5057 and 6067",
               "Los Angeles: 165 action groups",
               "Jinan: 911 action groups"],edge=BLUE,body_size=6.2)
        _node(ax,2.8,1.35,5.0,1.9,"Matched action labels",
              ["Same state; alternative phases",
               r"$z_g(a)=[Y_g(a)-Y_g(r_g)]/\sigma_g$"],edge=BLUE,
              body_size=6.3)

        _node(ax,8.8,6.25,5.15,2.8,"One source-group component",
              ["One source group + target groups",
               "Reference model: 29 features",
               "Pair model: 23 + 40 features",
               r"$S_g=(1-\alpha_g)A_g+\alpha_gP_g$",
               r"Jinan: $\alpha_g=0.9$; LA: local rule"],edge=ORANGE,
              body_size=6.1)
        _node(ax,8.8,3.45,5.15,2.1,"Target anchor",
              ["117-feature target-dominated model",
               "Domain-scaled cost contrasts",
               "Retained source prior for ties"],edge=ORANGE,body_size=6.1)
        _node(ax,8.8,1.25,5.15,1.85,"Development selection",
              ["8 separate closed-loop seeds",
               "Source identity → weight → guard",
               "Freeze before evaluation"],edge=ORANGE,body_size=6.0)

        _node(ax,15.0,6.35,5.35,2.6,"Selected city configurations",
              ["Los Angeles: target anchor",
               "Source weight 0; no learned guard",
               "Jinan: Atlanta; source weight 1",
               "Error-score risk multiplier 0.5"],edge=TEAL,body_size=6.1)
        _node(ax,15.0,3.65,5.35,2.45,"Score and guard the phase",
              [r"$\widehat a=\arg\min_a S_g(a)$",
               r"Jinan: $S_g(\widehat a)+0.5u_g(\widehat a)<0$",
               "Normalized pressure deficit ≤ 1",
               "Otherwise use pressure reference"],edge=TEAL,body_size=6.1)
        _node(ax,15.0,1.2,5.35,1.95,"Coordinate and execute",
              ["Nonconflicting learned interventions",
               "Minimum intervention gap: 60 s",
               "10-s decisions; common executor"],edge=TEAL,body_size=6.1)

        # Offline connections remain in gutters rather than crossing text boxes.
        _arrow(ax,(5.3,6.45),(6.225,6.45),color=BLUE)
        _route(ax,[(5.3,3.75),(5.6,3.75),(5.6,5.25),(6.225,5.25)],color=BLUE)
        _arrow(ax,(5.3,3.45),(6.225,3.45),color=BLUE)
        _arrow(ax,(2.8,2.625),(2.8,2.30),color=BLUE)
        _route(ax,[(5.3,1.35),(5.78,1.35),(5.78,4.98),(6.225,4.98)],color=BLUE)
        _route(ax,[(11.375,6.25),(11.67,6.25),(11.67,1.25),(11.375,1.25)],color=ORANGE)
        _arrow(ax,(8.8,2.4),(8.8,2.175),color=ORANGE)
        _route(ax,[(11.375,1.25),(11.85,1.25),(11.85,6.35),(12.325,6.35)],color=ORANGE)
        _arrow(ax,(15.0,5.05),(15.0,4.875),color=TEAL)
        _arrow(ax,(15.0,2.425),(15.0,2.175),color=TEAL)
        fig.text(0.5,0.024,
                 "CFCMT V93 · Adaptation and development selection precede frozen evaluation",
                 ha="center",fontsize=6.4,color="#46535C")
        return fig


if __name__=="__main__":
    destination=Path(__file__).resolve().parent/"figures"
    destination.mkdir(parents=True,exist_ok=True)
    for stem,draw in [("fig1_mechanism_timeline",draw_timeline),
                      ("fig2_framework_overview",draw_framework)]:
        figure=draw()
        for extension in ("png","pdf","svg"):
            figure.savefig(destination/f"{stem}.{extension}",dpi=300,
                           bbox_inches="tight",pad_inches=0.06)
        plt.close(figure)
        print(f"Saved {stem}")
