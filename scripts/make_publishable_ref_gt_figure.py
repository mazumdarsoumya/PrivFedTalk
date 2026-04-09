from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

ROOT = Path("/home/vineet/PycharmProjects/PrivFedTalk")
FIGSRC = ROOT / "outputs" / "qualitative_sheet" / "figures"
OUTDIR = ROOT / "paper_figures" / "qualitative_ref_gt"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "client": "0N1oA9LUEc4",
        "gt_idx": 27,
        "ref": FIGSRC / "ref_0N1oA9LUEc4.png",
        "gt": FIGSRC / "gt_0N1oA9LUEc4_t27.png",
    },
    {
        "client": "0QbLI1U4cLE",
        "gt_idx": 17,
        "ref": FIGSRC / "ref_0QbLI1U4cLE.png",
        "gt": FIGSRC / "gt_0QbLI1U4cLE_t17.png",
    },
    {
        "client": "13rqtiAPISY",
        "gt_idx": 26,
        "ref": FIGSRC / "ref_13rqtiAPISY.png",
        "gt": FIGSRC / "gt_13rqtiAPISY_t26.png",
    },
]

for r in rows:
    for k in ["ref", "gt"]:
        if not r[k].exists():
            raise FileNotFoundError(f"Missing image: {r[k]}")

cols = ["Input Reference", "Ground Truth"]

cell_w = 224
cell_h = 224
pad = 18
header_h = 54
row_text_h = 26
caption_h = 82

canvas_w = pad + 2 * (cell_w + pad)
canvas_h = header_h + len(rows) * (row_text_h + cell_h + pad) + caption_h + pad

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

font_header = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
font_row = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
font_caption = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)

# column headers
for j, col in enumerate(cols):
    x = pad + j * (cell_w + pad)
    bbox = draw.textbbox((0, 0), col, font=font_header)
    tw = bbox[2] - bbox[0]
    draw.text((x + (cell_w - tw) // 2, 14), col, fill="black", font=font_header)

# rows
for i, r in enumerate(rows):
    y0 = header_h + i * (row_text_h + cell_h + pad)
    row_label = f"Row {i+1}: {r['client']} | target frame {r['gt_idx']}"
    draw.text((pad, y0), row_label, fill="black", font=font_row)

    y = y0 + row_text_h
    paths = [r["ref"], r["gt"]]

    for j, p in enumerate(paths):
        x = pad + j * (cell_w + pad)
        tile = Image.open(p).convert("RGB").resize((cell_w, cell_h))
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(90, 90, 90), width=1)

caption = (
    "Reference and ground-truth frames for three selected identities from the LRS3 test split. "
    "For each row, the reference image is taken from frame index 0 of the selected clip, and the "
    "ground-truth image is taken from the matched target frame used for qualitative evaluation."
)
wrapped = textwrap.fill(caption, width=90)
draw.text((pad, canvas_h - caption_h + 8), wrapped, fill="black", font=font_caption)

png_out = OUTDIR / "qualitative_ref_gt.png"
canvas.save(png_out)

tex = r"""\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{paper_figures/qualitative_ref_gt/qualitative_ref_gt.png}
    \caption{Reference and ground-truth frames for three selected identities from the LRS3 test split. For each row, the reference image is taken from frame index 0 of the selected clip, and the ground-truth image is taken from the matched target frame used for qualitative evaluation.}
    \label{fig:qualitative_ref_gt}
\end{figure}
"""
tex_out = OUTDIR / "qualitative_ref_gt.tex"
tex_out.write_text(tex)

print("Saved PNG :", png_out)
print("Saved TEX :", tex_out)
