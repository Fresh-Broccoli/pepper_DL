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
    shutdown_server()
    return jsonify({
        'msg': 'success',
    })

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()



if __name__ == '__main__':
    # start flask app
    app.run(host="0.0.0.0", port=5000)
    del camera_manager
    del motion_manager
    del life_service
    del speech_manager
