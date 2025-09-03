import os 
import glob
import xml.etree.ElementTree as ET

annotations_dir = "./dataset/images/train"
images_dir = "./dataset/images/train"
labels_output_dir = "./dataset/labels/train"

os.makedirs(labels_output_dir, exist_ok=True)

class_map = {"tablet": 0}

def convert_bbox(size, box):
    """
    size    = (width, height)
    box     = (xmin, xmax, ymin, ymax)
    return x_center, y_center, w, h    
    """
    
    dw = 1. / size[0]
    dh = 1. / size[1]
    xmin, xmax, ymin, ymax = box

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    x_center = x_center * dw
    y_center = y_center * dh
    w = w * dw
    h = h * dh
    
    return x_center, y_center, w, h

for xml_file in glob.glob(os.path.join(annotations_dir, "*.xml")):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find("filename").text
    image_id = os.path.splitext(filename)[0]
    
    
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    label_path = os.path.join(labels_output_dir, f"{image_id}.txt")
    with open(label_path, "w") as f:
        for obj in  root.findall("object"):
            cls = obj.find("name").text
            if cls not in class_map:
                continue
            cls_id = class_map[cls]
            
            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)
            ymin = int(xmlbox.find("ymin").text)
            xmax = int(xmlbox.find("xmax").text)
            ymax = int(xmlbox.find("ymax").text)
            
            b = (xmin, xmax, ymin, ymax)
            bb = convert_bbox((w, h), b)
            
            f.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")
    print(f"[OK] {xml_file} -> {label_path}")

