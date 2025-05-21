import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import yaml
from ultralytics.utils.downloads import download

# Load the yaml configuration (defining class names and dataset paths)
yaml_file = "VOC.yaml"  # Ensure this file exists with the correct format
with open(yaml_file, 'r') as f:
    yaml = yaml.safe_load(f)

def convert_label(path, lb_path, year, image_id):
    """Converts XML annotations from VOC format to YOLO format by extracting bounding boxes and class IDs."""
    def convert_box(size, box):
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    in_file = open(path / f"VOC{year}/Annotations/{image_id}.xml")
    out_file = open(lb_path, "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    names = list(yaml["names"].values())  # names list
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls in names and int(obj.find("difficult").text) != 1:
            xmlbox = obj.find("bndbox")
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")

# Download
dir = Path(yaml["path"])  # dataset root dir
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
urls = [
    f"{url}VOCtrainval_11-May-2012.zip",  # 1.95GB, 17126 images
]
download(urls, dir=dir / "images", curl=True, threads=3, exist_ok=True)  # download and unzip over existing (required)

# Convert
path = dir / "images/VOCdevkit"
for year, image_set in ("2012", "train"), ("2012", "val"):
    imgs_path = dir / "images" / f"{image_set}{year}"
    lbs_path = dir / "labels" / f"{image_set}{year}"
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)
    with open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
        image_ids = f.read().strip().split()
    for id in tqdm(image_ids, desc=f"{image_set}{year}"):
        f = path / f"VOC{year}/JPEGImages/{id}.jpg"  # old img path
        lb_path = (lbs_path / f.name).with_suffix(".txt")  # new label path
        f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format