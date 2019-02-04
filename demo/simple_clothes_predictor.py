# Modified from maskrcnn-benchmark/demo/predictor.py

import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list

class ClothesDemo(object):
    CATEGORIES = [
        "__background__",
        "bag",
        "bottom",
        "one_piece",
        "shoe",
        "tops"
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        cfg = self.cfg

        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)

        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        image = self.transforms(original_image)
        image = image.to(self.device)
        image = image.unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(image)

        predictions = [o.to(self.cpu_device) for o in predictions]

        prediction = predictions[0]

        height, width = original_image.shape[:-1]

        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        colors = labels[:, None] * 10 * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        ksize = (10, 10)
        blurred = cv2.blur(image.copy(), ksize)

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple([top_left[0]+1, top_left[1]+1]), tuple([bottom_right[0]-1, bottom_right[1]-1]), tuple(color), 2
            )
            blurred[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        image = blurred

        return image

    def overlay_class_names(self, image, predictions):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return image
