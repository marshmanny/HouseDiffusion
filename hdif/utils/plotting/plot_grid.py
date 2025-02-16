import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pprint import pprint

tr = transforms.PILToTensor()

TSIZE = 0
SHAPE = 256
BORDER = 15
FACTOR = 2

def add_names_to_header(img, names, font=None, SHAPE=256):
    if font is None:
        font = ImageFont.truetype("/home/mdnikolaev/aikarpova_1/hd/hdif/utils/fonts/Times.ttf", 120 // (1024 // SHAPE))
    
    draw = ImageDraw.Draw(img)
    W_, H_ = img.size
    
    ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in names])
    new_img = Image.new('RGB', (W_, H_ + max(hw) + 100 // (1024 // SHAPE)), (255, 255, 255))

    draw = ImageDraw.Draw(new_img)
    W_, H_ = new_img.size
    new_img.paste(img, box=(0, max(hw) + 100 // (1024 // SHAPE)))
    
    for i, name in enumerate(names):
        draw.text((SHAPE * i + (SHAPE-ws[i])/2, max(hw) - min(hw) + 50 // (1024 // SHAPE)), name, font=font, fill='black')

    return new_img

#################################### 2 setup ################################################

if __name__ == "__main__":
    method2folder = {
        "run-20250130_065821-5oaapjv3": "/home/mdnikolaev/aikarpova_1/hdif_trainer/output/rplan_square_inf/run-20250130_065821-5oaapjv3/img/",
        "run-20250130_065822-uyho4qjk": "/home/mdnikolaev/aikarpova_1/hdif_trainer/output/rplan_square_inf/run-20250130_065822-uyho4qjk/img/",
    }

    file_patterns = {}

    file_patterns["square"] = {
        "run-20250130_065821-5oaapjv3": lambda size_id, img_id: f"square/{size_id}/{img_id}.png",
        "run-20250130_065822-uyho4qjk": lambda size_id, img_id: f"square/{size_id}/{img_id}.png",
    }

    file_patterns["pred"] = {
        "run-20250130_065821-5oaapjv3": lambda img_id: f"pred/{img_id}.png",
        "run-20250130_065822-uyho4qjk": lambda img_id: f"pred/{img_id}.png",
    }

    file_patterns["gt"] = {
        "run-20250130_065821-5oaapjv3": lambda img_id: f"gt/{img_id}.png",
        "run-20250130_065822-uyho4qjk": lambda img_id: f"gt/{img_id}.png",
    }

    preprocess = {
    }

    rename_methods = {
        "run-20250130_065821-5oaapjv3": "model=msd",
        "run-20250130_065822-uyho4qjk": "model=hd"
    }

    base_path = method2folder["run-20250130_065821-5oaapjv3"] + "square/0"
    sizes = list(sorted(os.listdir(method2folder["run-20250130_065821-5oaapjv3"] + "square")))

    files = os.listdir(base_path)
    exps = []
    for f in files:
        if ".png" in f or ".jpg" in f:
            exps.append(f.split('.')[0])

    exps = list(sorted(exps, key=lambda x: int(x)))[:5]
    methods = ["run-20250130_065821-5oaapjv3", "run-20250130_065822-uyho4qjk"]

    pprint(exps)

    new_img = Image.new('RGB', (SHAPE * (2 + len(sizes)), SHAPE * len(exps) * len(methods)), (255, 255, 255))

    offset = (0, -SHAPE)

    row_models = []
    for i, exp in enumerate(exps):
        for ii, method in enumerate(methods):
            offset = (0, offset[1] + SHAPE)
            path = method2folder[method] + file_patterns["gt"][method](exp)
            img_gt = Image.open(path).resize((SHAPE, SHAPE))
            new_img.paste(img_gt, box=offset)

            offset = (offset[0] + SHAPE, offset[1])
            path = method2folder[method] + file_patterns["pred"][method](exp)
            img_pred = Image.open(path).resize((SHAPE, SHAPE))
            new_img.paste(img_pred, box=offset)
            
            for j, size in enumerate(sizes):
                path = method2folder[method] + file_patterns["square"][method](size, exp)
                if os.path.exists(path):
                    img_s = Image.open(path)
                    if method in preprocess:
                        img_s = preprocess[method](img_s)
                    
                    img_s = img_s.resize((SHAPE, SHAPE))
                    
                    offset = (offset[0] + SHAPE, offset[1])
                    new_img.paste(img_s, box=offset)
                else:
                    print(path)
            
            row_models.append(rename_methods[method])

    img = new_img

    font = ImageFont.truetype("/home/mdnikolaev/aikarpova_1/hdif_trainer/hdif/utils/fonts/Times.ttf", 120 // (1024 // SHAPE)) 
    img = add_names_to_header(img, ["gt", "pred"] + [f'size={int(x) + 1}' for x in sizes], font, SHAPE)
    img = add_names_to_header(img.rotate(-90, expand=True), row_models[::-1], font, SHAPE)
    img = img.rotate(90, expand=True)

    img.save(f"/home/mdnikolaev/aikarpova_1/hdif_trainer/output/rplan_square_inf/grids/data_msd_grid1.png")
    img.save(f"/home/mdnikolaev/aikarpova_1/hdif_trainer/output/rplan_square_inf/grids/data_msd_grid1.pdf")
