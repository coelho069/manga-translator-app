from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((40, 40, 860, 1160), outline="black", width=5)
    draw.rectangle((70, 70, 830, 540), fill=(235, 235, 235), outline="black", width=3)
    draw.rectangle((70, 590, 830, 1130), fill=(245, 245, 245), outline="black", width=3)

    font = load_font(34)
    small_font = load_font(28)

    bubble1 = (160, 130, 520, 310)
    draw.ellipse(bubble1, fill="white", outline="black", width=4)
    draw.polygon([(440, 285), (500, 350), (420, 305)], fill="white", outline="black")
    draw.multiline_text((235, 185), "HELLO!\nHOW ARE YOU?", fill="black", font=font, spacing=8, align="center")

    bubble2 = (340, 700, 760, 930)
    draw.ellipse(bubble2, fill="white", outline="black", width=4)
    draw.polygon([(430, 895), (360, 980), (470, 920)], fill="white", outline="black")
    draw.multiline_text((455, 770), "I AM READY\nFOR THE NEXT\nADVENTURE!", fill="black", font=small_font, spacing=8, align="center")

    output_path = output_dir / "sample_manga_page.png"
    image.save(output_path)
    print(f"Imagem de exemplo criada em: {output_path}")


if __name__ == "__main__":
    main()

