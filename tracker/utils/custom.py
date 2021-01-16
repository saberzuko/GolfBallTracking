# import the necessary packages
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