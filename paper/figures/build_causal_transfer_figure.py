from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
BASE = HERE / "causal_transfer_gpt_image_base.png"
OUT = HERE / "causal_transfer_comparison.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=5)
    return box[2] - box[0], box[3] - box[1]


def label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    fnt: ImageFont.FreeTypeFont,
    pad: int = 12,
    radius: int = 12,
) -> None:
    x, y = xy
    w, h = text_size(draw, text, fnt)
    draw.rounded_rectangle(
        (x, y, x + w + 2 * pad, y + h + 2 * pad),
        radius=radius,
        fill=fill,
        outline=outline,
        width=2,
    )
    draw.multiline_text((x + pad, y + pad), text, fill=(28, 34, 42), font=fnt, spacing=5)


def centered_title(draw: ImageDraw.ImageDraw, panel: tuple[int, int], y: int, text: str, color: tuple[int, int, int]) -> None:
    fnt = font(29, bold=True)
    w, _ = text_size(draw, text, fnt)
    x = panel[0] + (panel[1] - panel[0] - w) // 2
    draw.text((x, y), text, fill=color, font=fnt)


def marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y = center
    r = 18
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 245), outline=color + (230,), width=3)
    fnt = font(20, bold=True)
    w, h = text_size(draw, text, fnt)
    draw.text((x - w / 2, y - h / 2 - 2), text, fill=color, font=fnt)


def main() -> None:
    base = Image.open(BASE).convert("RGBA")
    yoff = 84
    img = Image.new("RGBA", (1536, 1310), (255, 255, 255, 255))
    img.alpha_composite(base, (0, yoff))
    draw = ImageDraw.Draw(img, "RGBA")
    legend = font(18)

    panels = [(0, 512), (512, 1024), (1024, 1536)]
    centered_title(draw, panels[0], 26, "(a) H2O+-style dense residual", (33, 74, 132))
    centered_title(draw, panels[1], 26, "(b) MC-MC latent adaptation", (32, 118, 123))
    centered_title(draw, panels[2], 26, "(c) CFCMT causal transfer", (32, 112, 72))

    # Keep the generated visual layer readable: use small numbered anchors in the
    # image and put the explanatory text in a bottom legend.
    marker(draw, (336, yoff + 455), "1", (33, 74, 132))
    marker(draw, (672, yoff + 490), "2", (32, 118, 123))
    marker(draw, (1372, yoff + 520), "3", (32, 112, 72))

    draw.rounded_rectangle((36, 1126, 500, 1286), radius=12, fill=(248, 251, 255, 245), outline=(42, 99, 180, 210), width=2)
    draw.multiline_text(
        (58, 1148),
        "1  Dense residual block\n"
        "   One correction mixes headway,\n"
        "   passengers, dwell, reward,\n"
        "   and speed shifts across cities.",
        fill=(28, 34, 42),
        font=legend,
        spacing=5,
    )
    draw.rounded_rectangle((536, 1126, 1000, 1286), radius=12, fill=(246, 255, 255, 245), outline=(20, 151, 151, 210), width=2)
    draw.multiline_text(
        (558, 1148),
        "2  Latent world-model adaptation\n"
        "   Domain shift is compressed\n"
        "   into an implicit state; transit\n"
        "   mechanisms are not audited.",
        fill=(28, 34, 42),
        font=legend,
        spacing=5,
    )
    draw.rounded_rectangle((1036, 1126, 1500, 1286), radius=12, fill=(248, 255, 248, 245), outline=(42, 135, 80, 210), width=2)
    draw.multiline_text(
        (1058, 1148),
        "3  CFCMT causal transfer\n"
        "   Unlabeled target summaries\n"
        "   gate source evidence; separated\n"
        "   mechanisms remain auditable.",
        fill=(28, 34, 42),
        font=legend,
        spacing=5,
    )

    img.convert("RGB").save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
