import sys
import numpy as np
import os
import cv2
import time
import requests
import io
import base64
from PIL import Image

# SORT:
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "sort"))
#from trackers.sort.sort import SortManager

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager

# BoTSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "botsort"))
from trackers.botsort.bot_sort import *

models = {
    #"sort":SortManager,
    "ocsort":OCSortManager,
    "botsort":BoTSortManager,
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
        self.vertical_ratio = None
        self.horizontal_ratio = None
        self.last_box = None
        print(f"Loading {model}...")
        self.dl_model = models[model](**kwargs)
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
        #pred = self.dl_model.update(img)
        if draw:
            self.draw(pred, img, save_dir="images")
        return pred, img

    def draw(self, prediction, img, show=None, save_dir=None):
        self.dl_model.draw(prediction, np.ascontiguousarray(img), show=show, save_dir=save_dir)

    def follow_behaviour(self):
        self.stop()
        try:
            while True:
                self.rotate_head_abs()
                ctarget_id = self.dl_model.target_id
                if self.dl_model.target_id != self.dl_model.max_target_id:
                    self.spin(speed=0.1)
                pred, img = self.predict(img=None, draw=False)
                print("Prediction:", pred)
                if ctarget_id == 0:
                    if ctarget_id != self.dl_model.target_id :
                        self.stop()
                        self.say("Target detected")
                else:
                    if ctarget_id != self.dl_model.target_id:
                        self.stop()
                        self.say("Target Lost")
                #print("Length of pred: ", len(pred))
                self.center_target(pred, img.shape, )
                self.last_box = pred
        except Exception as e:
            print(e)
            self.shutdown()

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
            # If the length of box is zero, that means Pepper just lost track of the target before it officially
            # declares the target lost. In this window, we can still recover the track by making Pepper move towards
            # wherever the target could've shifted to
            if self.vertical_ratio is not None and self.horizontal_ratio is not None and self.dl_model.target_id!=0:
                self.walkToward(theta=self.horizontal_ratio*1.5)

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

            # Saves a copy of the last ratio
            self.vertical_ratio = vertical_ratio
            self.horizontal_ratio = horizontal_ratio

            if abs(horizontal_ratio) > stop_threshold:
                # If horizontal ratio is not within the stop threshold, rotate to center the target
                self.walkToward(theta=horizontal_ratio*0.9)
            else:
                # Otherwise, approach target
                self.approach_target(box, img_shape, commands=["rotate_head"],commands_kwargs=[{"forward":vertical_ratio*0.2}])

    def approach_target(self, box, img_shape, stop_threshold=0.65, move_back_threshold=0.8, commands=None, commands_kwargs=None):
        # (x1, y1, x2, y2, id)
        box_area = (box[2]-box[0])*(box[3]-box[1])
        frame_area = img_shape[0]*img_shape[1]
        ratio = box_area/frame_area
        if ratio >= stop_threshold:
            # Add end condition here:
            self.stop()
            self.say("Hello, I'm Pepper, do you require my assistance?")
            self.dl_model.reset_trackers()
            """
            if ratio > move_back_threshold:
                self.walkTo(x=(ratio-1)/3)
            else:
                if commands is not None: # assumes that commands is a list
                    for i in range(len(commands)):
                        self.robot_actions[commands[i]](**commands_kwargs[i])
            """
        else:
            self.walkToward(x=1-ratio, y=self.horizontal_ratio*ratio)

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
            pred, img = self.predict(None, draw=False)
            end_time = time.time() - start_time
        print(f"It took {str(end_time)} seconds to receive {str(image_no)} images. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")

def dummy_action():
    # does nothing
    pass

def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)

if __name__ == "__main__":

    def ocfollow():
        c = Client(model="ocsort", image_size=[640,640], device="cuda", max_age=60, verbose=False, hand_raise_frames_thresh=3)
        #c = Client(image_size=[640, 640], device="cpu", max_age=60, verbose=True)
        # Main follow behaviour:
        c.follow_behaviour()
        # Must call
        c.shutdown()

    # BoTSORT default params
    args = make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score

    def botfollow():
        #c = Client(model="botsort", image_size=[640,640], device="cuda", max_age=60, verbose=True, hand_raise_frames_thresh=3)
        c = Client(model="botsort", image_size=[640,640], device="cuda", verbose=False, args=args, hand_raise_frames_thresh=3)
        # Main follow behaviour:
        c.follow_behaviour()
        # Must call
        c.shutdown()

    def livestream_camera_botsort():
        c = Client(model="botsort", image_size=[640, 640], device="cuda", verbose=False, args=args,
                   hand_raise_frames_thresh=3)
        vertical_offset = 0.5
        try:
            while True:
                pred, img = c.predict(img=None, draw=False)
                if len(pred) > 0:
                    box = pred[0]
                    img_shape = img.shape
                    box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                        3] * vertical_offset])  # box[1]/2+box[3]/2])
                    frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                    # diff = box_center - frame_center
                    diff = frame_center - box_center
                    horizontal_ratio = diff[0] / img_shape[1]
                    vertical_ratio = diff[1] / img_shape[0]
                    area = (box[2]-box[0])*(box[3]-box[1])
                    area_ratio = area/(img_shape[0]*img_shape[1])
                    print("BoT Prediction:", pred)
                    print("Area ratio:", area_ratio)
                    print("horizontal_ratio:", horizontal_ratio)
                    print("vertical_ratio:", vertical_ratio)

        except Exception as e:
            print(e)
            c.shutdown()

    def livestream_camera_ocsort():
        c = Client(model="ocsort", image_size=[640,640], device="cuda", max_age=60, verbose=False, hand_raise_frames_thresh=3)
        vertical_offset = 0.5
        try:
            while True:
                pred, img = c.predict(img=None, draw=False)
                if len(pred) > 0:
                    box = pred[0]
                    img_shape = img.shape
                    box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                        3] * vertical_offset])  # box[1]/2+box[3]/2])
                    frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                    # diff = box_center - frame_center
                    diff = frame_center - box_center
                    horizontal_ratio = diff[0] / img_shape[1]
                    vertical_ratio = diff[1] / img_shape[0]
                    area = (box[2]-box[0])*(box[3]-box[1])
                    area_ratio = area/(img_shape[0]*img_shape[1])
                    print("Prediction:", pred)
                    print("Area ratio:", area_ratio)
                    print("horizontal_ratio:", horizontal_ratio)
                    print("vertical_ratio:", vertical_ratio)

        except Exception as e:
            print(e)
            c.shutdown()

    #c = Client(image_size=[640,640], device="cuda", max_age=60, use_byte=True, verbose=True)
    #c = Client(image_size=[640, 640], device="cpu", max_age=60, use_byte=True, verbose=True)
    #c = Client(model="ocsort", device="cuda", use_byte=True, verbose=True)
    #c = Client(model="botsort", device="cuda", verbose=True, args=args)
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
    #c.get_image_pred_test(60)
    #c.pepper_to_server_fps()

    # Main follow behaviour:
    #c.follow_behaviour()

    # Must call
    #c.shutdown()
    #livestream_camera_ocsort()
    #livestream_camera_botsort()
    botfollow()

    # Call to quickly shut down without creating an instance of Client
    quick_shutdown()