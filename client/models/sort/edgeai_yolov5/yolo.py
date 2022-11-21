import torch
import os
import cv2
import numpy as np
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import colors, plot_one_box

class YoloManager():
    def __init__(self, weights='yoloposes_640_lite.pt', device="cpu", save_txt_tidl=True, image_size=[640,640], kpt_label=True):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA
        weights_dir = os.path.join(os.path.dirname(__file__), "weights", weights)
        # Load model
        self.model = attempt_load(weights_dir, map_location=self.device)  # load model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.kpt_label = kpt_label

        if isinstance(image_size, (list,tuple)):
            assert len(image_size) ==2; "height and width of image has to be specified"
            image_size[0] = check_img_size(image_size[0], s=self.stride)
            image_size[1] = check_img_size(image_size[1], s=self.stride)
        else:
            image_size = check_img_size(image_size, s=self.stride)  # check img_size
        self.image_size = image_size
        if self.half:
            self.model.half()


    def preprocess_frame(self, frame):
        # Preprocesses a frame that's in numpy array format to be compatible with self.model
        # Padded resize

        img = letterbox(frame, self.image_size, stride=self.stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #print(img.shape)
        return img

    def extract_bounding_box_data(self, prediction):
        return prediction[:, :5]

    def extract_keypoint_data(self, prediction):
        # Extracts prediction keypoints and reshapes them into:
        #   (number of detections, 17 key points, 3 features consisting of x, y, confidence)
        return prediction[:, 6:].reshape(prediction.shape[0], 17,3)

    def extract_bounding_box_and_keypoint(self, prediction):
        return self.extract_bounding_box_data(prediction), self.extract_keypoint_data(prediction)

    def predict(self, frame, augment=False, conf_thres=0.25, classes=None, iou_thres=0.45, agnostic_nms=False, preprocess=True, scale_to_original=True):
        if preprocess:
            original_shape = frame.shape
            frame = self.preprocess_frame(frame)
        pred = self.model(frame, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=self.kpt_label)[0]
        if scale_to_original:
            self.scale_to_original(pred, original_shape)
        return pred

    def scale_to_original(self, prediction, original_shape):
        scale_coords(self.image_size, prediction[:, :4], original_shape, kpt_label=False)
        scale_coords(self.image_size, prediction[:, 6:], original_shape, kpt_label=self.kpt_label, step=3)

    def draw(self, prediction):
        for det_index, (*xyxy, conf, cls) in enumerate(reversed(prediction[:,:6])):
            plot_one_box(xyxy, img, label=(f'{self.names[int(cls)]} {conf:.2f}'), color=colors(int(cls),True), line_thickness=2, kpt_label=self.kpt_label, kpts=prediction[det_index, 6:], steps=3, orig_shape=img.shape[:2])
        cv2.imshow("Image", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    from PIL import Image

    manager = YoloManager(image_size=[640,640])

    img = cv2.imread(os.path.join("data", "custom", "forest.jpg"))

    preprocessed_img = manager.preprocess_frame(img)
    pred = manager.predict(img, conf_thres=0.50, scale_to_original=True)
    #box, point = manager.extract_bounding_box_and_keypoint(pred)

    # Visualising
    manager.draw(pred)
