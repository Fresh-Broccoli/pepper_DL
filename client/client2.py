import socket
import sys
import pickle
import numpy as np
import os
import cv2
import time
import requests
import io
import base64
from PIL import Image

# SORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "sort"))
from trackers.sort.sort import SortManager

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager

models = {
    "sort":SortManager,
    "ocsort":OCSortManager,
}


class Client:
    def __init__(self, model="ocsort", address='http://localhost:5000', **kwargs):
        self.address = address
        print(f"Loading {model}...")
        self.dl_model = models[model](use_byte=True, **kwargs)
        print(model, " loaded successfully!")

    def get_image(self,show=False, save=False, save_name=None):
        headers = {'content-type':"/image/send_image"}
        response = requests.post(self.address+headers["content-type"], headers=headers)
        j = response.json()
        img = np.array(Image.open(io.BytesIO(base64.b64decode(j['img']))))[:,:,::-1] # Convert from BGR to RGB
        if show:
            cv2.imshow("Pepper_Image", img)
            cv2.waitKey(1)
        if save:
            cv2.imwrite(f"images/{save_name.png}", img)
        return img

    def predict(self, img, draw=True):
        if img is None:
            img = self.get_image()
        # Shape of pred: number of tracked targets x 5
        # where 5 represents: (x1, y1, x2, y2, id)
        pred = self.dl_model.smart_update(img)

        if draw:
            self.draw(pred, img, save_dir="images")
        return pred, img

    def draw(self, prediction, img, show=None, save_dir=None):
        self.dl_model.draw(prediction, np.ascontiguousarray(img), show=show, save_dir=save_dir)

    def get_image_test(self, image_no=60):
        start_time = time.time()
        for i in range(image_no):
            self.get_image(save=True, save_name=str(i))
        end_time = time.time() - start_time
        print(f"It took {str(end_time)} seconds to receive {str(image_no)} images. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")

    def pepper_to_server_fps(self, ):
        headers = {'content-type':"/test/pepper_to_server_fps"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        j = response.json()
        print(
            f"It took {str(j['time'])} seconds to receive {str(j['frames'])} images. This means we were able to receive images from Pepper to server to client at {str(float(j['frames']) / float(j['time']))} FPS!")

    def get_image_pred_test(self, image_no=60):
        start_time = time.time()
        for i in range(image_no):
            pred, img = self.predict(None, draw=True)
        end_time = time.time() - start_time
        print(f"It took {str(end_time)} seconds to receive {str(image_no)} images. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")

    def shutdown(self):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)


if __name__ == "__main__":
    c = Client(image_size=[640,640], device="cuda", max_age=45)
    #c = Client(image_size=[640, 640], device="cpu")

    #img = c.get_image(show=False)
    #cv2.imwrite(f"images/test_{time.time()}.png", img)

    c.get_image_pred_test(30)
    #c.pepper_to_server_fps()
    c.shutdown()