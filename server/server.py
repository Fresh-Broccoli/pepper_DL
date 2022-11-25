import socket
import os
import sys

from PIL import Image
import numpy as np
import qi
import cPickle
import time
from camera import CameraManager
from locomotion import MovementManager
from threading import Timer

# Copied from: https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

class Server:
    def __init__(self, camera_manager, motion_manager, port=5000):
        self.host = socket.gethostname()
        self.port = port
        self.camera_manager = camera_manager
        self.motion_manager = motion_manager
        self.server_socket = socket.socket()
        self.server_socket.bind((self.host, self.port))
        print "Listening..."
        self.server_socket.listen(2)
        self.conn, self.address = self.server_socket.accept()
        print "Connection from: " + str(self.address)

    def communicate(self):
        while True:
            action = raw_input("Please choose an action: ")
            self.conn.send(action.encode())
            if action == "send image":
                start_time = time.time()
                p = os.path.join(os.getcwd(), "imgs", "dog.jpg")
                img = np.asarray(Image.open(p))
                data = cPickle.dumps(img)
                print "Sent data is ", sys.getsizeof(data), " bytes."
                self.conn.send(data)
                d = self.conn.recv(1024)
                print "It took " + str(time.time() - start_time) + " seconds to load, transmit, show, and receive confirmation."

            elif action == "send pepper":
                start_time = time.time()
                self.send_image()
                d = self.conn.recv(1024)
                #conn.send(cPickle.dumps(raw_image[0]))
                #conn.send(cPickle.dumps(raw_image[1]))
                print "It took " + str(time.time() - start_time) + " seconds to take a picture, transmit to ML server, and receive confirmation"

            elif action == "pepper pred":
                start_time = time.time()
                self.send_image()
                pred = self.conn.recv(10000)
                print "Received prediction size = ", sys.getsizeof(pred), " bytes."
                #pred = cPickle.loads(pred, encoding='latin1')
                #d = conn.recv(1024)
                #conn.send(cPickle.dumps(raw_image[0]))
                #conn.send(cPickle.dumps(raw_image[1]))
                print "It took " + str(time.time() - start_time) + " seconds to take a picture, transmit to ML server, make prediction and receive confirmation"

            elif action == "livestream":
                fps = None
                duration = None
                pfps = cPickle.dumps(fps)
                pduration = cPickle.dumps(duration)
                print "Pickled fps = ", sys.getsizeof(pfps), " bytes."
                print "Pickled fps = ", sys.getsizeof(pduration), " bytes."
                self.conn.send(pfps)
                self.conn.send(pduration)
                self.livestream(fps, duration)

            elif action == "bye":
                break

        self.conn.close()  # close the connection

    def send_image(self):
        raw_image = self.camera_manager.get_image()
        data = cPickle.dumps(raw_image)
        print "Sent data is ", sys.getsizeof(data), " bytes."
        self.conn.send(data)

    def livestream(self, fps=10, duration=None):
        wait = 1/fps if fps is not None else 0
        if duration:
            count = 0
            t_end = time.time() + duration
            while time.time() < t_end:
                if fps:
                    time.sleep(wait)
                self.send_image()
                self.conn.recv(8)
                count += wait

        else:
            while True:
                time.sleep(wait)
                self.send_image()
                m = self.conn.recv(8).decode()
                if m == "b":
                    break


if __name__ == '__main__':

    session = qi.Session()

    ip = "192.168.137.248"
    port = 9559

    try:
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) + ".\n"
                                                                             "Please check your script arguments. Run with -h option for help.")
    life_service = session.service("ALAutonomousLife")
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    life_service.setAutonomousAbilityEnabled("All", False) # Disable autonomous life

    # First, wake up.
    motion_service.wakeUp()

    fractionMaxSpeed = 0.8
    # Go to posture stand
    posture_service.goToPosture("StandInit", fractionMaxSpeed)



    camera_manager = CameraManager(session, resolution=1, colorspace=11, fps=12)
    motion_manager = MovementManager(session)

    s = Server(camera_manager)
    s.communicate()
    #server_program()

    del camera_manager
    del motion_manager
    #motion_service.rest()
    #posture_service.goToPosture("Sit", fractionMaxSpeed)