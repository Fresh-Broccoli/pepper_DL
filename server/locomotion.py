import qi
import argparse
import sys
import time
import os

class MovementManager:
    def __init__(self, session):
        self.motion_service  = session.service("ALMotion")
        self.posture_service = session.service("ALRobotPosture")
        self.motion_service.setStiffnesses("Head", 1.0)
        self.reset_posture()

    def reset_posture(self, speed=0.5):
        """ Makes the robot stand up straight. Used to reset posture.
        Params:

        speed: float
            Influences the speed in which the robot adopts the specified default posture. Ranges from 0 to 1.
        """
        # Wake up robot
        self.motion_service.wakeUp()
        # Make the robot stand up straight.
        self.posture_service.goToPosture("StandInit", speed)

    def walkToward(self, x=0, y=0, theta=0, verbose=False):
        """ Makes the robot walk at speeds relative to its maximum speed until its collision avoidance or the user
            causes it to stop. This function will not interrupt the main flow of the program, however calling this
            function multiple times will terminate older executions.

        Params:

        x: float
            Controls the speed in which the robot moves foward. Ranges from -1 to 1.
        y: float
            Controls the speed in which the robot strafes left. Ranges from -1 to 1.
        theta: float
            Controls the speed in which the robot rotates anti-clockwise.
        verbose: bool
            Determines whether we want to print out parameters and results.
        """
        if verbose:
            print "Walking with forward=", x, "m, left=", y, "m, rotation=", theta, "degrees anti-clockwise..."
        self.motion_service.moveToward(x,y,theta)

    def stop(self):
        """ Stops walking

        """
        self.motion_service.stopMove()


    def rotate_head(self, forward=0, left=0, speed=0.2):
        """ Rotates Pepper's head

        param:

        forward: float
            Controls the extent of moving Pepper's head forward. Ranges from -1 to 1.
        left: float
            Controls the extent of moving
        speed: float

        """
        self.motion_service.setAngles(["HeadYaw", "HeadPitch"], [left,forward], speed)

    def terminate(self, fractionMaxSpeed=0.2):
        # Shuts down services upon deletion
        self.posture_service.goToPosture("Sit", fractionMaxSpeed)
        self.motion_service.rest()

    def __del__(self):
        # Shuts down services upon deletion
        self.posture_service.goToPosture("Sit", 0.2)
        self.motion_service.rest()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.137.210",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                                                                                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    manager = MovementManager(session)

    # del manager