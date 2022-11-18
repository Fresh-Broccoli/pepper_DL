import socket
import sys
import pickle
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


#sys.path.append(os.path.join(os.getcwd(), "models", "edgeai_yolov5"))

from yolo import YoloManager

# Copied from: https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

class Client:
    def __init__(self, port=5000, host=None, model="yolo", **kwargs):
        self.host = socket.gethostname() #if host is not None else host
        self.port = port
        self.client_socket = socket.socket()

        if model == "yolo":
            self.dl_model = YoloManager(**kwargs)

        self.client_socket.connect((self.host, self.port))

    # Each model might require a unique function for configuration because they accept different parameters
    #def configure_yolo_model(self, weights='yoloposes_640_lite.pt', image_size=640, save_txt_tidle=True, device="cpu"):
    #    self.dl_model = YoloManager(weights=weights,  image_size=image_size, save_txt_tidle=save_txt_tidle, device=device)

    #def connect(self):
    #    self.client_socket.connect((self.host, self.port))

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
                img = np.asarray(real_img)
                #break
                #print("data: ", img)
                #print("data type is ", type(img))
                #print("shape: ", img.shape)

                pred = self.dl_model.predict(img)
                #print("pred type: ", type(pred))
                #print(pred)
                #break
                self.dl_model.draw(pred, img)
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
                    #fig = plt.figure()
                    #creating a subplot
                    #ax1 = fig.add_subplot(1,1,1)

                    while time.time() < t_end:
                        data = self.client_socket.recv(300000)  # receive response
                        print("Received data is ", sys.getsizeof(data), " bytes.")
                        #print('Received from server: ' , data)  # show in terminal
                        data = pickle.loads(data, encoding='latin1')
                        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                        img = np.asarray(real_img)
                        #pred = self.dl_model.predict(img)

                        # Stream live data
                        cv2.imshow("stream", img)
                        cv2.waitKey(1)
                        self.client_socket.send("a".encode())
                        count += wait
                        frames += 1

                    end_time = time.time()-start
                    print("FPS test for input=80x60 resolution wireless:")
                    print("It took ", end_time, " seconds to stream ", frames, " frames.")
                    #print("count: ", count)
                    print("fps: ", fps)
                    #print("duration: ", duration)
                    print("true fps: ", frames/end_time)

                else:
                    while True:
                        data = self.client_socket.recv(300000)  # receive response
                        print("Received data is ", sys.getsizeof(data), " bytes.")
                        #print('Received from server: ' , data)  # show in terminal
                        data = pickle.loads(data, encoding='latin1')
                        real_img = Image.frombuffer('RGB', (data[0], data[1]), bytes(data[6]))
                        img = np.asarray(real_img)
                        pred = self.dl_model.predict(img)
                        self.client_socket.send("a".encode())



            elif server_action == "bye":
                break
            print("It took ", time.time()-start, " seconds to process this request.")
        self.client_socket.close()  # close the connection
        #return real_img, pred
if __name__ == '__main__':
    c = Client(image_size=[640,640])
    # img, pred =
    c.communicate()