import socket
import os
import sys

import numpy as np
import qi
import cPickle
import time
import argparse
import io
import base64

from flask import Flask, request, Response, jsonify
from PIL import Image
from camera import CameraManager
from locomotion import MovementManager
from voice import SpeechManager
from threading import Timer
from ast import literal_eval
# Copied from: https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client

class Server:
    def __init__(self, camera_manager, motion_manager, speech_manager, port=5000):
        self.host = socket.gethostname()
        self.port = port
        self.camera_manager = camera_manager
        self.motion_manager = motion_manager
        self.speech_manager = speech_manager
        self.server_socket = socket.socket()
        self.server_socket.bind((self.host, self.port))

        self.child_functions = {
            "walkToward": self.motion_manager.walkToward,
            "walkTo": self.motion_manager.walkTo,
            "rotate_head": self.motion_manager.rotate_head,
            "rotate_head_abs": self.motion_manager.rotate_head_abs,
            "target_lost": self.speech_manager.target_lost,
            "target_detected": self.speech_manager.target_detected,
            "stop": self.motion_manager.stop,

        }

        print "Listening..."
        self.server_socket.listen(1)
        self.conn, self.address = self.server_socket.accept()
        print "Connection from: " + str(self.address)
        self.speech_manager.say("Connected to deep learning client. Please raise your hand if you want me to follow you.")

    def neo_communicate(self):
        while True:
            #fps = None
            #duration = None
            #pfps = cPickle.dumps(fps)
            #pduration = cPickle.dumps(duration)
            #print "Pickled fps = ", sys.getsizeof(pfps), " bytes."
            #print "Pickled fps = ", sys.getsizeof(pduration), " bytes."
            #self.conn.send(pfps)
            #self.conn.send(pduration)

            self.livestream(None, False)

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
                #m = self.conn.recv(8).decode()
                commands = self.receive_message()
                #code, func, params = command.split("|")
                messages = commands.split("$")
                actions = [command.split("|") for command in messages[1:]]
                #message = command.split("|")
                print("actions = ", actions)
                #print("message = ", message)
                if messages[0] == "b":
                    break
                elif messages[0] == "c":
                    # center target
                    # Parsing arguments came from: https://stackoverflow.com/questions/9305387/string-of-kwargs-to-kwargs
                    for action in actions:
                        self.child_functions[action[0]](**dict((k, literal_eval(v)) for k, v in (pair.split('=') for pair in action[1].split())))
                    #self.child_functions[message[1]](**dict((k, literal_eval(v)) for k, v in (pair.split('=') for pair in message[2].split())))

    def receive_message(self, bts=1024):
        """ Receives encoded message from client and outputs the decoded text
        Params:
            bts: int
                represents how many bytes of data the client is ready to receive from the server
        Returns:
            string representing the message
        """
        return self.conn.recv(bts).decode()

# Initiate Qi Session
session = qi.Session()
parser = argparse.ArgumentParser(description="Please enter Pepper's IP address (and optional port number)")
parser.add_argument("--ip", type=str, nargs='?', default="192.168.43.183")
parser.add_argument("--port", type=int, nargs='?', default=9559)
args = parser.parse_args()

print("Received IP: ", args.ip)
print("Received port: ", args.port)
# Connecting to the robot
# Decided not to use try-except because if the server failed to connect to the robot, then there's no point
    # starting a Flask server
session.connect("tcp://" + args.ip + ":" + str(args.port))

print("Connected to Pepper!")

# Robot setup:
# Controls autonomous life behaviour, mainly used to disable it
print("Subscribing to live service...")
life_service = session.service("ALAutonomousLife")
# Controls the robot's cameras
print("Subscribing to camera service...")
camera_manager = CameraManager(session, resolution=1, colorspace=11, fps=30)
# Controls the robot's locomotion
print("Subscribing to movement service...")
motion_manager = MovementManager(session)
# Controls the robot's speech
print("Subscribing to speech service...")
speech_manager = SpeechManager(session)

# Disable autonomous life because it can interfere with follow behaviour
life_service.setAutonomousAbilityEnabled("All", False)


# Start Flask server
app = Flask(__name__)

@app.route("/image/send_image", methods=["POST"])
def send_image():
    # Requests image from Pepper, then sends it to the client
    img = camera_manager.get_image(raw=False) # Gets image as Pillow Image
    rawBytes = io.BytesIO()
    # Saves image as buffer
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({
        'msg': 'success',
        'img': str(img_base64),
    })

@app.route("/voice/startup_greeting", methods=["POST"])
def startup_greeting():
    # Makes Pepper greet the user and provide basic instructions
    speech_manager.say("Connected to deep learning client. Please raise your hand if you want me to follow you.")


@app.route("/test/pepper_to_server_fps", methods=["POST"])
def pepper_to_server_fps():
    start = time.time()
    frames = 60
    for _ in range(frames):
        img = camera_manager.get_image(raw=False)  # Gets image as Pillow Image
        rawBytes = io.BytesIO()
        # Saves image as buffer
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        img = np.array(Image.open(io.BytesIO(base64.b64decode(img_base64))))
    end = time.time() - start
    return jsonify({
        "time":str(end),
        "frames":str(frames)
    })
    print "It took " + str(end) + " seconds to send " + str(frames) + " frames at " + str(frames/end) + "FPS."

@app.route("/setup/end", methods=["POST"])
def shutdown():
    # Run to shut down
    speech_manager.say("Shutting down")
    del camera_manager
    del motion_manager
    del life_service
    del speech_manager
    return jsonify({
        'msg': 'success',
    })


if __name__ == '__main__':
    # start flask app
    app.run(host="0.0.0.0", port=5000)

