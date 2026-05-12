from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent / "bubbles"


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_multiline(draw: ImageDraw.ImageDraw, box, text: str, font, fill="black", spacing=8) -> None:
    lines = text.splitlines()
    measurements = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bbox[2] - bbox[0] for bbox in measurements]
    heights = [bbox[3] - bbox[1] for bbox in measurements]
    total_height = sum(heights) + spacing * max(0, len(lines) - 1)
    x1, y1, x2, y2 = box
    y = y1 + ((y2 - y1) - total_height) // 2
    for line, width, height in zip(lines, widths, heights):
        x = x1 + ((x2 - x1) - width) // 2
        draw.text((x, y), line, fill=fill, font=font)
        y += height + spacing


def draw_tail(draw: ImageDraw.ImageDraw, points, fill="white", outline="black", width=4) -> None:
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def draw_ellipse_bubble(draw: ImageDraw.ImageDraw, box, tail=None, width=4) -> None:
    draw.ellipse(box, fill="white", outline="black", width=width)
    if tail:
        draw_tail(draw, tail, width=width)


def draw_rounded_bubble(draw: ImageDraw.ImageDraw, box, radius=34, tail=None, width=4) -> None:
    draw.rounded_rectangle(box, radius=radius, fill="white", outline="black", width=width)
    if tail:
        draw_tail(draw, tail, width=width)


def add_light_noise(image: Image.Image, amount: int = 900, seed: int = 7) -> None:
    rng = random.Random(seed)
    pixels = image.load()
    width, height = image.size
    for _ in range(amount):
        x = rng.randrange(width)
        y = rng.randrange(height)
        gray = rng.randrange(205, 245)
        pixels[x, y] = (gray, gray, gray)


def save_simple_bubble_01() -> None:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(42)
    box = (115, 95, 525, 315)
    draw_ellipse_bubble(draw, box, tail=[(390, 292), (455, 390), (345, 310)])
    centered_multiline(draw, box, "HELLO!\nHOW ARE YOU?", font)
    image.save(OUTPUT_DIR / "simple_bubble_01.png")


def save_simple_bubble_02() -> None:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(38)
    box = (90, 80, 550, 335)
    draw_rounded_bubble(draw, box, radius=48, tail=[(170, 316), (130, 405), (240, 330)])
    centered_multiline(draw, box, "I FOUND\nA SECRET DOOR!", font)
    image.save(OUTPUT_DIR / "simple_bubble_02.png")


def save_simple_bubble_03() -> None:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(34)
    box = (145, 70, 505, 360)
    draw_ellipse_bubble(draw, box, tail=[(470, 260), (550, 345), (440, 305)])
    centered_multiline(draw, box, "WAIT...\nTHIS IS\nDANGEROUS!", font)
    image.save(OUTPUT_DIR / "simple_bubble_03.png")


def save_long_text_bubble() -> None:
    image = Image.new("RGB", (820, 560), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(30)
    box = (90, 65, 730, 405)
    draw_rounded_bubble(draw, box, radius=62, tail=[(530, 385), (620, 510), (480, 415)])
    centered_multiline(
        draw,
        box,
        "I HAVE BEEN WAITING\nFOR THIS MOMENT\nSINCE THE DAY\nWE FIRST MET!",
        font,
        spacing=7,
    )
    image.save(OUTPUT_DIR / "long_text_bubble.png")


def save_small_bubble() -> None:
    image = Image.new("RGB", (420, 320), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(28)
    box = (105, 80, 315, 205)
    draw_ellipse_bubble(draw, box, tail=[(250, 190), (295, 260), (218, 205)], width=3)
    centered_multiline(draw, box, "NO\nWAY!", font, spacing=4)
    image.save(OUTPUT_DIR / "small_bubble.png")


def save_double_bubble() -> None:
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    font_a = load_font(32)
    font_b = load_font(30)
    box_a = (70, 80, 390, 285)
    box_b = (500, 185, 830, 410)
    draw_ellipse_bubble(draw, box_a, tail=[(285, 265), (350, 350), (250, 285)])
    draw_rounded_bubble(draw, box_b, radius=44, tail=[(585, 388), (525, 495), (650, 410)])
    centered_multiline(draw, box_a, "WHERE\nARE WE?", font_a)
    centered_multiline(draw, box_b, "WE ARE\nINSIDE THE\nCASTLE.", font_b, spacing=6)
    image.save(OUTPUT_DIR / "double_bubble.png")


def save_noisy_bubble() -> None:
    image = Image.new("RGB", (720, 520), "white")
    add_light_noise(image, amount=2200)
    draw = ImageDraw.Draw(image)
    font = load_font(32)
    box = (95, 80, 625, 370)
    draw_ellipse_bubble(draw, box, tail=[(430, 350), (515, 470), (382, 382)])
    centered_multiline(draw, box, "THE SIGNAL\nIS GETTING\nWEAKER!", font, spacing=6)
    image.save(OUTPUT_DIR / "noisy_bubble.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_simple_bubble_01()
    save_simple_bubble_02()
    save_simple_bubble_03()
    save_long_text_bubble()
    save_small_bubble()
    save_double_bubble()
    save_noisy_bubble()
    print(f"Imagens de exemplo criadas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
