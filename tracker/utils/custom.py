# import the necessary packages
from scipy.spatial import distance
from glob import glob
import os
import cv2

# the function is used to calculate the moemnts and return the center x and y co-ordinates
def findCentre(contours):
    moments = cv2.moments(contours)
    cX = int(moments["m10"]/moments["m00"])
    cY = int(moments["m01"]/moments["m00"])
    return (cX, cY)

# The function warpPerspectiveCustom is used to transform the co-ordinates from the 
# original image (p) to the perspective transformed image co-ordinates using the projection matrix
def warpPerspectiveCustom(p, projectionMatrix):
    x = (projectionMatrix[0][0]*p[0] + projectionMatrix[0][1]*p[1] + projectionMatrix[0][2]) / ((projectionMatrix[2][0]*p[0] + projectionMatrix[2][1]*p[1] + projectionMatrix[2][2]))
    y = (projectionMatrix[1][0]*p[0] + projectionMatrix[1][1]*p[1] + projectionMatrix[1][2]) / ((projectionMatrix[2][0]*p[0] + projectionMatrix[2][1]*p[1] + projectionMatrix[2][2]))
    return (int(round(x)), int(round(y)))

# function to list videos in the specified path
def listVideos(PID):
    path = os.path.join("./database", str(PID), "*.mov")
    videos = glob(path)
    return videos

# This class handles the creation of the CSV file and updating the CSV files in 
# the output directory
class CSV_Writer():
    # csv file is created as soon as an instance of the class is created
    def __init__(self, PID):
        self.fileName = str(PID)+".csv"
        self.csvPath = "./output/"
        with open(self.csvPath + self.fileName, "w") as writer:
            writer.write("participant_id,SID,putt_id,speed,angle,make")
            writer.write("\n")
    
    # function for updating the csv file
    def csvUpdate(self, PID=None, SID=None, PtID=None, speed=None, 
    angle=None, make=None, processing=True):
        if processing == True:
            with open(self.csvPath + self.fileName, "a") as writer:
                writer.write(PID + "," + SID + "," + PtID + "," + str(speed) + "," + str(angle) + "," + str(make))
                writer.write("\n")
        else:
            with open(self.csvPath + self.fileName, "a") as writer:
                writer.write(PID + "," + SID + "," + PtID + "," + "NaN" + "," + "NaN" + "," + "NaN")
                writer.write("\n")

# This class handles the ball dynamics: to calculate the velocity of the ball, 
# the angle of the ball before entering the hole and the make (entered hole or not)
class BallDynamics():
    # initializing the velocity start and stop coordinates
    def __init__(self, velocityStart, velocityStop, fps, holerect):
        self.velocityStart = velocityStart
        self.velocityStop = velocityStop
        self.fps = fps
        self.hole_x = holerect[0]
        self.hole_y = holerect[1]
        self.hole_w = holerect[2]
        self.hole_h = holerect[3]
        self.velTrackFlag = True
        self.destReachedFlag = False
        self.initReachedFlag = False
        self.ballinFlag = False
        self.frameCounter = 0
        self.initX, self.initY = None, None
        self.destX, self.destY = None, None
    
    # function is used to calculate the velocity of the ball
    def velocityCal(self, ballXWarped, ballYWarped):

        # if the ball is in the given limits of the tracking co-ordinates and the tracking flag is true
        # then the frame counter will increment
        if (self.velocityStart <= ballYWarped <= self.velocityStop) and self.velTrackFlag:
            self.frameCounter += 1

            # once we enter the outer if branch it indicates we have started calculating the velocity
            # if we have entered the limits for the first time then we memorise the ball co-ordniates
            # and set the initReachedFlag to true
            if not self.initReachedFlag:
                self.initReachedFlag = True
                self.initX, self.initY = ballXWarped, ballYWarped
        
        # if the ball is greater than the bounds of the tracking co-oridnates and we have crossed the bounds
        # for the first time we track the co-ordinates of the ball and we stop tracking by setting the flag
        # velTrackFlag to false
        if (ballYWarped > self.velocityStop) and not self.destReachedFlag:
            self.velTrackFlag = False
            self.destReachedFlag = True
            self.destX, self.destY = ballXWarped, ballYWarped

        if self.initReachedFlag and self.destReachedFlag:
            distanceTravelled = distance.euclidean((self.initX, self.initY), (self.destX, self.destY))
            timeTravelled = self.frameCounter/float(self.fps)
            ballPixelVelocity = distanceTravelled/float(timeTravelled)
            return(ballPixelVelocity)
        else:
            return None
    
    # function used to see if the ball entered the hole or not
    def ballMake(self, ballX, ballY):

        # if the ballinFlag is false then check if the ball has entered the hole or not and update the flag accordingly
        if not self.ballinFlag and (self.hole_x <= ballX <= self.hole_x + self.hole_w) and (self.hole_y <= ballY <= self.hole_y + self.hole_h):
            self.ballinFlag = True
        # if ballinFlag is true then check if the ball is outside the hole or not and update the flag accordingly
        if self.ballinFlag and (self.hole_x > ballX) or (ballX > self.hole_x + self.hole_w) or (self.hole_y > ballY) or (ballY > self.hole_y + self.hole_h):
            self.ballinFlag = False
        return self.ballinFlag
