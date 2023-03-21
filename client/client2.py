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
        if show: # Don't use if running remotely
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

    def follow_behaviour(self):
        pass

    def center_target(self, box, img_shape, stop_threshold = 0.1, vertical_offset=0.5, lost = None):
        """ Takes in target bounding box data and attempts to center it
        Preconditons:
            1. box must contain data about exactly 1 bounding box

        Params:
            box: 2D array
                Data about one bounding box in the shape of: 1 x 5
                Columns must be in the format of (x1, y1, x2, y2, id)
            img_shape: 1D array
                Shape of the original frame in the format: (height, width, colour channels),
                so passing in original_image.shape will do.
            stop_threshold: float between 0 and 1
                If the difference between box center and frame center over frame resolution is less than this threshold,
                tell the robot to stop rotating, otherwise, the robot will be told to rotate at a rate proportional to
                this ratio.
            vertical_offset: float between 0 and 1


        Returns:
            a string in the format a|b|c, where:
                a = message code, tells the server what mode it should be running (or what kind of data to expect)
                b = function the server will call
                c = parameter in the format of param_name=param param_name2=param2 param_name3=param3 ...
                        this part is also optional, because functions that b is referring to might not require params
        """
        if len(box)!=1:
            #raise Exception(f"The length of box is {len(box)}, but it should be 1!")
            #return "c$stop|"
            #return "c$stop|" + "$say|text=\"Target lost, searching for new target\"" if (len(box)==0 and self.dl_model.target_id is None) else "" + "$rotate_head_abs|"
            return "c$stop|" + "$target_lost|" if lost=="l" else "" + "$rotate_head_abs|"
            #return "c$stop|" + "$rotate_head_abs|"
        if len(img_shape)!=3:
            raise Exception(f"The shape of the image does not equal to 3!")

        # Since there's an extra dimension, we'll take the first element, which is just the single detection
        box = box[0]

        # Following shapes will be (x, y) format
        box_center = np.array([box[2]/2+box[0]/2, box[1]*(1-vertical_offset)+box[3]*vertical_offset])#box[1]/2+box[3]/2])
        frame_center = np.array((img_shape[1]/2, img_shape[0]/2))
        #diff = box_center - frame_center
        diff = frame_center - box_center
        horizontal_ratio = diff[0]/img_shape[1]
        vertical_ratio = diff[1]/img_shape[0]

        if abs(horizontal_ratio) > stop_threshold:
            #print("ratio = ", horizontal_ratio)
            # difference ratio greater than threshold, rotate at that ratio
            # locomotion_manager.walkToward(theta=horizontal_ratio)
            o = f"c$walkToward|theta={str(horizontal_ratio*0.9)}"
            #return f"c$walkTo|theta={str(horizontal_ratio*0.9)}"
        else:
            #return "c$stop|"
            o = self.approach_target(box, img_shape, command=f"rotate_head|forward={str(vertical_ratio*0.2)}")
            print("o = ", o)
        return o[0:2] + ("target_detected|$" if lost=="t" else "") +o[2:]

    def say(self, word):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)


    def shutdown(self):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)

    #------------------------------------------------------------------------------------------------------------------
    # Test code #######################################################################################################
    #------------------------------------------------------------------------------------------------------------------
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


if __name__ == "__main__":
    c = Client(image_size=[640,640], device="cuda", max_age=45)
    #c = Client(image_size=[640, 640], device="cpu")

    #img = c.get_image(show=False)
    #cv2.imwrite(f"images/test_{time.time()}.png", img)

    c.say("I'm Pepper, I like eating pizza with pineapple")
    c.say("I am not a fan of hamburgers with fish and tomato sauce")
    #c.get_image_pred_test(30)
    #c.pepper_to_server_fps()
    c.shutdown()