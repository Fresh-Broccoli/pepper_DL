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

    def __init__(self, model="ocsort", address='http://localhost:5000', verbose=False, **kwargs):
        self.address = address
        self.robot_actions = {
            "walkToward": self.walkToward,
            "walkTo": self.walkTo,
            "rotate_head": self.rotate_head,
            "rotate_head_abs": self.rotate_head_abs,
            "say": self.say,
            "target_lost": self.target_lost,
            "target_detected": self.target_detected,
        }
        self.verbose = verbose
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

        try:
            while True:
                if self.dl_model.target_id != self.dl_model.max_target_id:
                    self.spin(speed=0.1)
                pred, img = self.predict(img=None, draw=False)
                self.center_target(pred, img.shape, )
        except Exception as e:
            print(e)






    def center_target(self, box, img_shape, stop_threshold = 0.1, vertical_offset=0.5):
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
        """
        if len(img_shape)!=3: # Check shape of image
            raise Exception(f"The shape of the image does not equal to 3!")

        if len(box)>1: # Check number of tracks
            # If not 1, then the target is either lost, or went off-screen
            #raise Exception(f"The length of box is {len(box)}, but it should be 1!")
            self.stop()
            if self.dl_model.target_id == 0:
                print("Target Lost")
                #self.target_lost()
            else:
                self.rotate_head_abs()
        elif len(box) == 0:
            pass
        else: # If there's only 1 track, center the camera on them
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
                self.walkToward(theta=horizontal_ratio*0.9)
            else:
                self.approach_target(box, img_shape, commands=["rotate_head"],commands_kwargs=[{"forward":vertical_ratio*0.2}])

    def approach_target(self, box, img_shape, stop_threshold=0.65, move_back_threshold=0.8, commands=None, commands_kwargs=None):
        # (x1, y1, x2, y2, id)
        box_area = (box[2]-box[0])*(box[3]-box[1])
        frame_area = img_shape[0]*img_shape[1]
        ratio = box_area/frame_area
        if ratio > stop_threshold:
            # Add end condition here:

            if ratio > move_back_threshold:
                self.walkTo(x=(ratio-1)/2)
            else:
                if commands is not None: # assumes that commands is a list
                    for i in range(len(commands)):
                        self.robot_actions[commands[i]](**commands_kwargs[i])
        else:
            self.walkToward(x=1-ratio)

    def spin(self, left=True, speed=0.2, verbose = False):
        self.walkToward(theta = speed if left else -speed, verbose=verbose)
        # Keep resetting head position
        self.rotate_head_abs()

    #-------------------------------------------------------------------------------------------------------------------
    # Robot controls ###################################################################################################
    #-------------------------------------------------------------------------------------------------------------------

    def say(self, word, verbose = False):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)
        if verbose or self.verbose:
            print(f"say(word={word})")

    def target_lost(self, verbose = False):
        headers = {'content-type': "/voice/targetLost"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose or self.verbose:
            print(f"target_lost()")

    def target_detected(self, verbose = False):
        headers = {'content-type': "/voice/targetDetected"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose or self.verbose:
            print(f"target_detected()")

    def stop(self, verbose = False):
        headers = {'content-type': "/locomotion/stop"}
        response = requests.post(self.address + headers["content-type"])
        if verbose or self.verbose:
            print(f"stop()")

    def walkTo(self, x=0, y=0, theta=0, verbose=False):
        headers = {'content-type': "/locomotion/walkTo"}
        response = requests.post(self.address + headers["content-type"] + f"?x={str(x)}&y={str(y)}&theta={str(theta)}&verbose={str(1 if verbose else 0)}")
        if verbose or self.verbose:
            print(f"walkTo(x={str(x)}, y={str(y)}, theta={str(theta)})")

    def walkToward(self, x=0, y=0, theta=0, verbose=False):
        headers = {'content-type': "/locomotion/walkToward"}
        response = requests.post(self.address + headers["content-type"] + f"?x={str(x)}&y={str(y)}&theta={str(theta)}&verbose={str(1 if verbose else 0)}")
        if verbose or self.verbose:
            print(f"walkToward(x={str(x)}, y={str(y)}, theta={str(theta)})")

    def rotate_head(self, forward=0, left=0, speed=0.2, verbose=False):
        headers = {'content-type': "/locomotion/rotateHead"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        if verbose or self.verbose:
            print(f"rotate_head(forward={str(forward)}, left={str(left)}, speed={str(speed)})")

    def rotate_head_abs(self, forward=0, left=0, speed=0.2, verbose=False):
        headers = {'content-type': "/locomotion/rotateHeadAbs"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        if verbose or self.verbose:
            print(f"rotate_head_abs(forward={str(forward)}, left={str(left)}, speed={str(speed)})")


    def shutdown(self, verbose=False):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose or self.verbose:
            print("shutdown()")

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

def dummy_action():
    # does nothing
    pass

def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)

if __name__ == "__main__":
    #c = Client(image_size=[640,640], device="cuda", max_age=60, verbose=True)
    #c = Client(image_size=[640, 640], device="cpu")

    # Voice functions:
    #c.say("I'm Pepper, I like eating pizza with pineapple")
    #c.say("I am not a fan of hamburgers with fish and tomato sauce")

    # Locomotion functions:
    #c.walkTo(x=0.3)
    #c.walkTo(y=0.3)
    #c.walkTo(theta=0.3)
    #c.walkToward(x=0.2, theta=0.2)
    #c.rotate_head_abs(forward=1)
    #time.sleep(3)
    #c.rotate_head_abs(forward=-1)
    #time.sleep(3)
    #c.rotate_head_abs(left=1)
    #time.sleep(3)
    #c.rotate_head_abs(left=-1)
    #time.sleep(3)


    # Image test functions
    #c.get_image_pred_test(30)
    #c.pepper_to_server_fps()

    # Main follow behaviour:
    #c.follow_behaviour()

    # Must call
    #c.shutdown()
    quick_shutdown()