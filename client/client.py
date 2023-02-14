import socket
import sys
import pickle
import numpy as np
import os
import cv2
import time
#import matplotlib
#matplotlib.use('TKAgg')
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from PIL import Image

# Uncomment the following lines to use SORT or OCSORT

# SORT:
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "sort"))
#from trackers.sort.sort import SortManager

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager
# Copied from: https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

class Client:
    def __init__(self, port=5000, host=None, model="sort", **kwargs):
        self.host = socket.gethostname() #if host is not None else host
        self.port = port
        self.client_socket = socket.socket()
        print(f"Loading {model}...")
        #if model == "yolo":
        #    self.dl_model = YoloManager(**kwargs)
        #elif model == "sort":
        #    self.dl_model = SortManager(**kwargs)
        self.dl_model = OCSortManager(use_byte=True, **kwargs)
        print(model, " loaded successfully!")
        self.client_socket.connect((self.host, self.port))

    # Each model might require a unique function for configuration because they accept different parameters
    #def configure_yolo_model(self, weights='yoloposes_640_lite.pt', image_size=640, save_txt_tidle=True, device="cpu"):
    #    self.dl_model = YoloManager(weights=weights,  image_size=image_size, save_txt_tidle=save_txt_tidle, device=device)

    #def connect(self):
    #    self.client_socket.connect((self.host, self.port))
    def neo_communicate(self):
        while True:
            fps = pickle.loads(self.client_socket.recv(1000), encoding="latin1")
            duration = pickle.loads(self.client_socket.recv(1000), encoding="latin1")

            wait = 1/fps if fps is not None else 0
            if duration:
                count = 0
                start = time.time()
                t_end = start + duration
                frames = 0

                # Visualisation ini

                while time.time() < t_end:
                    data = self.client_socket.recv(300000)  # receive response
                    print("Received data is ", sys.getsizeof(data), " bytes.")
                    #print('Received from server: ' , data)  # show in terminal
                    data = pickle.loads(data, encoding='latin1')
                    real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                    img = np.asarray(real_img)[:,:,::-1]
                    print(img.shape)
                    #pred = self.dl_model.predict(img)

                    # Stream live data
                    cv2.imshow("stream", img)
                    cv2.waitKey(1)
                    self.client_socket.send("a".encode())
                    count += wait
                    frames += 1
                cv2.destroyAllWindows()
                end_time = time.time()-start
                print("FPS test for input=80x60 resolution wireless:")
                print("It took ", end_time, " seconds to stream ", frames, " frames.")
                #print("count: ", count)
                print("fps: ", fps)
                #print("duration: ", duration)
                print("true fps: ", frames/end_time)

            else:
                while True:
                    try:
                        data = self.client_socket.recv(300000)  # receive response
                        print("Received data is ", sys.getsizeof(data), " bytes.")
                        #print('Received from server: ' , data)  # show in terminal
                        data = pickle.loads(data, encoding='latin1')
                        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                        img = np.asarray(real_img)[:,:,::-1]
                        # Shape of the image: (height, width, colour channels)
                        pred = self.dl_model.update(img)
                        # Shape of pred: number of tracked targets x 5
                        # where 5 represents: (x1, y1, x2, y2, id)
                        self.dl_model.draw(pred, np.ascontiguousarray(img), show=1)
                        m = self.center_target(pred, img.shape)
                        self.client_socket.send(m.encode())
                    except KeyboardInterrupt:
                        self.client_socket.send("b".encode())
                        cv2.destroyAllWindows()

    def communicate(self):
        while True:

            server_action = self.client_socket.recv(2*1024).decode() # confirmation code
            start = time.time()
            if server_action == "send pepper":
                data = self.client_socket.recv(1000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                #print('Received from server: ' , data)  # show in terminal
                data = pickle.loads(data, encoding='latin1')
                real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                print("Received image shape = ", np.asarray(real_img).shape)
                #real_img = real_img.rotate(180)
                real_img.show()
                self.client_socket.send("d".encode())
                #print("type of data = ", type(data))
                # show numpy
            elif server_action == "pepper pred":
                data = self.client_socket.recv(2000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                #print('Received from server: ' , data)  # show in terminal
                data = pickle.loads(data, encoding='latin1')
                real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                img = np.asarray(real_img)[:,:,::-1]
                #break
                #print("data: ", img)
                #print("data type is ", type(img))
                #print("shape: ", img.shape)

                pred = self.dl_model.update(img)
                #print("pred type: ", type(pred))
                #print(pred)
                #break
                self.dl_model.draw(pred, np.ascontiguousarray(img), 0)
                #pred_dump = pickle.dumps(pred)
                #print("Sent prediction size = ", sys.getsizeof(pred_dump), " bytes.")
                self.client_socket.send("a".encode())

            elif server_action == "send image":
                data = self.client_socket.recv(2000000)  # receive response
                print("Received data is ", sys.getsizeof(data), " bytes.")
                data = pickle.loads(data, encoding="latin1")

                #img = Image.fromarray(data, "RGB")
                #img.show()
                self.client_socket.send("d".encode())
                #img = Image.frombytes("RGB", )
                # show dog image
                #plt.imshow(data)
                #plt.show()
                #print(type(data))

            elif server_action == "livestream":
                fps = pickle.loads(self.client_socket.recv(1000), encoding="latin1")
                duration = pickle.loads(self.client_socket.recv(1000), encoding="latin1")

                wait = 1/fps if fps is not None else 0
                if duration:
                    count = 0
                    start = time.time()
                    t_end = start + duration
                    frames = 0

                    # Visualisation ini

                    while time.time() < t_end:
                        data = self.client_socket.recv(300000)  # receive response
                        print("Received data is ", sys.getsizeof(data), " bytes.")
                        #print('Received from server: ' , data)  # show in terminal
                        data = pickle.loads(data, encoding='latin1')
                        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                        img = np.asarray(real_img)[:,:,::-1]
                        print(img.shape)
                        #pred = self.dl_model.predict(img)

                        # Stream live data
                        cv2.imshow("stream", img)
                        cv2.waitKey(1)
                        self.client_socket.send("a".encode())
                        count += wait
                        frames += 1
                    cv2.destroyAllWindows()
                    end_time = time.time()-start
                    print("FPS test for input=80x60 resolution wireless:")
                    print("It took ", end_time, " seconds to stream ", frames, " frames.")
                    #print("count: ", count)
                    print("fps: ", fps)
                    #print("duration: ", duration)
                    print("true fps: ", frames/end_time)

                else:
                    while True:
                        try:
                            data = self.client_socket.recv(300000)  # receive response
                            print("Received data is ", sys.getsizeof(data), " bytes.")
                            #print('Received from server: ' , data)  # show in terminal
                            data = pickle.loads(data, encoding='latin1')
                            real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                            img = np.asarray(real_img)[:,:,::-1]
                            #pred = self.dl_model.predict(img)
                            pred = self.dl_model.update(img)
                            self.dl_model.draw(pred, np.ascontiguousarray(img), show=1)
                            self.client_socket.send("a".encode())
                        except KeyboardInterrupt:
                            self.client_socket.send("b".encode())
                            cv2.destroyAllWindows()

            elif server_action == "bye":
                break

            print("It took ", time.time()-start, " seconds to process this request.")
        self.client_socket.close()  # close the connection
        #return real_img, pred

    def center_target(self, box, img_shape, stop_threshold = 0.1):
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
                if the difference between box center and frame center over frame resolution is less than this threshold,
                tell the robot to stop rotating, otherwise, the robot will be told to rotate at a rate proportional to
                this ratio.
        """
        if len(box)!=1:
            #raise Exception(f"The length of box is {len(box)}, but it should be 1!")
            return "c|stop|"
        if len(img_shape)!=3:
            raise Exception(f"The shape of the image does not equal to 3!")

        box = box[0]

        # Following shapes will be (x, y) format
        box_center = np.array([box[2]/2+box[0]/2, box[1]/2+box[3]/2])
        frame_center = np.array((img_shape[1]/2, img_shape[0]/2))
        #diff = box_center - frame_center
        diff = frame_center - box_center
        #diff = (box_center[0]-frame_center[0], box_center[1]-frame_center[1])
        horizontal_ratio = diff[0]/img_shape[1]
        #vertical_ratio = diff[1]/img_shape[0]
        if abs(horizontal_ratio) > stop_threshold:
            # difference ratio greater than threshold, rotate at that ratio
            # locomotion_manager.walkToward(theta=horizontal_ratio)
            return f"c|walkToward|theta={str(horizontal_ratio)}"
        else:
            #return "c|stop|"
            return self.approach_target(box, img_shape)

    def approach_target(self, box, img_shape):
        # (x1, y1, x2, y2, id)
        box_area = (box[2]-box[0])*(box[3]-box[1])
        frame_area = img_shape[0]*img_shape[1]
        ratio = box_area/frame_area
        return f"c|walkToward|x={str(1-ratio)}"

if __name__ == '__main__':
    c = Client(image_size=[640,640])
    c.neo_communicate()
    #c.communicate()