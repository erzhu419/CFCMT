from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
OUT = HERE / "validation_protocol_schematic.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    return draw.multiline_textbbox(xy, text, font=fnt, spacing=6)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    spacing: int = 6,
) -> None:
    x0, y0, x1, y1 = box
    tb = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_color: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 46), radius=18, fill=outline + (255,), outline=outline, width=0)
    draw.rectangle((x0, y0 + 24, x1, y0 + 46), fill=outline + (255,))
    title_font = font(22, bold=True)
    title_width = text_box(draw, (0, 0), title, title_font)[2]
    if title_width > (x1 - x0 - 28):
        title_font = font(19, bold=True)
    centered_text(draw, (x0 + 14, y0 + 4, x1 - 14, y0 + 44), title, title_font, (255, 255, 255))
    centered_text(draw, (x0 + 16, y0 + 56, x1 - 16, y1 - 12), body, font(18), title_color)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 5,
    dashed: bool = False,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if dashed:
        segments = 16
        for i in range(segments):
            if i % 2 == 0:
                xa = x0 + (x1 - x0) * i / segments
                ya = y0 + (y1 - y0) * i / segments
                xb = x0 + (x1 - x0) * (i + 1) / segments
                yb = y0 + (y1 - y0) * (i + 1) / segments
                draw.line((xa, ya, xb, yb), fill=color, width=width)
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    # Arrowhead.
    import math

    ang = math.atan2(y1 - y0, x1 - x0)
    size = 15
    left = (x1 - size * math.cos(ang - 0.5), y1 - size * math.sin(ang - 0.5))
    right = (x1 - size * math.cos(ang + 0.5), y1 - size * math.sin(ang + 0.5))
    draw.polygon([end, left, right], fill=color)


def routed_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 5,
    dashed: bool = False,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        arrow(draw, start, end, color, width=width, dashed=dashed)
        # Hide intermediate arrowheads to keep the route visually clean.
        draw.ellipse((end[0] - 12, end[1] - 12, end[0] + 12, end[1] + 12), fill=(255, 255, 255, 255))
    arrow(draw, points[-2], points[-1], color, width=width, dashed=dashed)


def lock(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.arc((x, y, x + 46, y + 48), 190, 350, fill=(178, 76, 53), width=5)
    draw.rounded_rectangle((x + 5, y + 28, x + 41, y + 68), radius=6, fill=(255, 244, 238), outline=(178, 76, 53), width=3)
    draw.ellipse((x + 20, y + 43, x + 26, y + 49), fill=(178, 76, 53))
    draw.rectangle((x + 22, y + 48, x + 24, y + 58), fill=(178, 76, 53))


def main() -> None:
    img = Image.new("RGBA", (1600, 920), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Palette.
    navy = (36, 78, 128)
    teal = (34, 125, 128)
    green = (42, 132, 82)
    orange = (210, 123, 42)
    red = (178, 76, 53)
    gray = (105, 115, 125)
    text = (30, 36, 44)

    centered_text(draw, (70, 22, 1530, 70), "Validation protocol and data-construction boundary", font(30, True), text)

    rounded_box(
        draw,
        (60, 118, 320, 300),
        "Open agency data",
        "GTFS routes/stops/trips\nPassenger volume or APC\nTraffic speed bands\nSchedule-derived proxies",
        (246, 250, 255),
        navy,
        text,
    )
    rounded_box(
        draw,
        (390, 118, 650, 300),
        "City bundle",
        "Full-route network\nRoute-direction-stop events\nTime and local descriptors\nEvidence-level audit fields",
        (247, 252, 255),
        teal,
        text,
    )
    rounded_box(
        draw,
        (704, 92, 996, 238),
        "Uncalibrated simulator",
        "f0(s,a)\nSimulator output only\nNo target calibration",
        (246, 250, 255),
        navy,
        text,
    )
    rounded_box(
        draw,
        (704, 274, 996, 420),
        "Static-derived target",
        "fc(s,a)\nObserved + apportioned +\nschedule-derived components",
        (255, 250, 243),
        orange,
        text,
    )
    rounded_box(
        draw,
        (1050, 146, 1330, 366),
        "Source residual labels",
        "Delta_c = fc(s,a) - f0(s,a)\nUsed only for source cities\nFits dense H2O+-style and\nCFCMT mechanism residuals",
        (248, 255, 248),
        green,
        text,
    )

    arrow(draw, (320, 209), (390, 209), teal)
    arrow(draw, (650, 190), (704, 165), navy)
    arrow(draw, (650, 230), (704, 340), orange)
    arrow(draw, (996, 165), (1050, 215), navy)
    arrow(draw, (996, 340), (1050, 295), orange)

    # Split line.
    draw.line((70, 468, 1530, 468), fill=(205, 210, 216), width=3)
    centered_text(draw, (70, 438, 1530, 462), "Strict leave-one-city-out validation", font(23, True), gray)

    rounded_box(
        draw,
        (90, 535, 360, 730),
        "Split",
        "Hold out one city as target\nTrain on all other cities\nEvaluation unit is city,\nnot route or repeated split",
        (246, 250, 255),
        navy,
        text,
    )
    rounded_box(
        draw,
        (430, 515, 705, 760),
        "Allowed target inputs",
        "Unlabeled target covariates\nSchedules and simulator outputs\nSpeed/demand summaries\nStatic local descriptors",
        (248, 255, 248),
        green,
        text,
    )
    rounded_box(
        draw,
        (775, 515, 1050, 760),
        "Blocked during fitting",
        "Target transition labels\nTarget residual labels\nTarget-route calibration\nTarget-label model selection",
        (255, 246, 244),
        red,
        text,
    )
    rounded_box(
        draw,
        (1120, 535, 1400, 730),
        "Final reporting",
        "Use held-out target labels\nonly after selection is frozen\nReport MSE ratios, mechanism\nerrors, and policy proxies",
        (255, 250, 243),
        orange,
        text,
    )

    arrow(draw, (360, 630), (430, 630), green)
    arrow(draw, (705, 630), (775, 630), red, dashed=True)
    arrow(draw, (1050, 630), (1120, 630), orange)
    lock(draw, 882, 782)
    centered_text(draw, (770, 838, 1060, 884), "No target labels for fitting or source selection", font(20, True), red)

    # Source training and allowed target summaries both feed the frozen evaluation.
    arrow(draw, (1190, 366), (1190, 535), green, width=4)
    routed_arrow(draw, [(568, 760), (568, 792), (1118, 792), (1118, 690)], green, width=4, dashed=True)

    # A small legend.
    draw.rounded_rectangle((60, 812, 560, 880), radius=12, fill=(248, 250, 252), outline=(190, 198, 206), width=2)
    draw.line((86, 835, 150, 835), fill=green, width=5)
    draw.text((166, 822), "available before target evaluation", font=font(18), fill=text)
    draw.line((86, 858, 150, 858), fill=red, width=5)
    draw.text((166, 845), "blocked until final reporting", font=font(18), fill=text)

    img.convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
