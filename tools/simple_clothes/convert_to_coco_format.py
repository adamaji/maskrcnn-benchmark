import json
import sys
import os
import csv
from PIL import Image
from random import shuffle

CLASSES = (
    "__background__",
    "bag",
    "bottom",
    "one_piece",
    "shoe",
    "tops"
)

class_to_index = dict(zip(CLASSES, range(len(CLASSES))))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python convert.py [path_to_garments_csv] [output path]")
        sys.exit(-1)

    fn = sys.argv[1]
    output = sys.argv[2]

    rows = []
    with open(fn, newline='') as cf:
        reader = csv.reader(cf)
        for row in reader:
            rows.append(row)

    images = []
    images_seen = []
    for row in rows:
        url = row[0]
        image_id = int(url.split("/")[-1].split(".jpg")[0])
        if image_id not in images_seen:
            path = "./simple_clothes/images/%s.jpg" % str(image_id)
            image = Image.open(path)
            width, height = image.size
            images.append({
                'license': 0,
                'file_name': "%s.jpg" % str(image_id),
                'coco_url': url,
                'height': int(height),
                'width': int(width),
                'date_captured': '',
                'flickr_url': url,
                'id': image_id
            })
            images_seen.append(image_id)

    shuffle(images_seen)
    train_split = images_seen[:int(len(images_seen)*0.6)]
    val_split = images_seen[int(len(images_seen)*0.6):int(len(images_seen)*0.8)]
    test_split = images_seen[int(len(images_seen)*0.8):]
    images_train = [img for img in images if img['id'] in train_split]
    images_val = [img for img in images if img['id'] in val_split]
    images_test = [img for img in images if img['id'] in test_split]

    annotations = []
    anno_count = 0
    for row in rows:
        url, name, xmin, ymin, _, _, xmax, ymax, _, _ = row
        image_id = url.split("/")[-1].split(".jpg")[0]
        path = "./simple_clothes/images/%s.jpg" % str(image_id)
        image = Image.open(path)
        width, height = image.size
        xmin = int(float(xmin) * width)
        ymin = int(float(ymin) * height)
        xmax = int(float(xmax) * width)
        ymax = int(float(ymax) * height)
        category_id = class_to_index[name]
        annotations.append({
            'segmentation': [],
            'area': (xmax - xmin) * (ymax - ymin),
            'iscrowd': 0,
            'image_id': int(image_id),
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'category_id': int(category_id),
            'id': anno_count
        })
        anno_count += 1
    annos_train = [anno for anno in annotations if anno['image_id'] in train_split]
    annos_val = [anno for anno in annotations if anno['image_id'] in val_split]
    annos_test = [anno for anno in annotations if anno['image_id'] in test_split]

    data = {}
    data['info'] = {}
    data['images'] = []
    data['licenses'] = []
    data['annotations'] = []
    data['categories'] = []

    # categories
    for cls in class_to_index:
        if cls != "__background__":
            data['categories'].append({
                'supercategory': 'garment',
                'id': class_to_index[cls],
                'name': cls
            }) 

    data_train = data.copy()
    data_train['images'] = images_train
    data_train['annotations'] = annos_train

    data_val = data.copy()
    data_val['images'] = images_val
    data_val['annotations'] = annos_val

    data_test = data.copy()
    data_test['images'] = images_test
    data_test['annotations'] = annos_test

    print("Writing to %s" % output)
    with open(os.path.join(output, "train.json"), "w") as f:
        json.dump(data_train, f)
    with open(os.path.join(output, "val.json"), "w") as f:
        json.dump(data_val, f)
    with open(os.path.join(output, "test.json"), "w") as f:
        json.dump(data_test, f)
