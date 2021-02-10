# import the necessary packages
from tracker.utils.custom import warpPerspectiveCustom
from tracker.utils.ballDetection import BallDetector
from tracker.utils.intialFrame import FrameHandling
from tracker.utils.custom import BallDynamics
from tracker.utils.custom import CSV_Writer
from tracker.utils.custom import listVideos
from tracker.utils.custom import findCentre
from progress.bar import IncrementalBar
from tracker.conf.conf import Conf
from scipy.spatial import distance
import numpy as np
import cv2
import os

# reading the json configuration file
conf = Conf("tracker/conf/confg.json")

# assigning the fourcc for video writing
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# accessing each PID from the json file
for PID in conf["PIDs"]:
    # instantiating the CSV_Writer, which will create a CSV file in the output directory
    csv = CSV_Writer(PID)
    
    # creating the directory for storing the processed and warped videos
    output_path = os.path.join("./output", str(PID))
    processed_output_path = os.path.join(output_path, "processed")
    warped_output_path = os.path.join(output_path, "warped")
    os.mkdir(output_path)
    os.mkdir(processed_output_path)
    os.mkdir(warped_output_path)
    
    # listing all the videos associated with the PID
    videos = listVideos(PID)

    #creating a progress bar
    bar = IncrementalBar("Countdown", max = len(videos))

    for video in videos:
        bar.next()
        video_name = video[video.rfind("/")+1:].split(".")[0] + ".avi"
        ids = video[video.rfind("/")+1:].split("_")
        SID = ids[1]
        PtID = ids[2].split(".")[0]
        vidcap = cv2.VideoCapture(video)
        success, frame = vidcap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]/1.25), int(frame.shape[0]/1.25)))
        
        # creating an instance of the FrameHandling class to detect the green area and the hole in the frame
        frameHandler = FrameHandling(frame, conf["greenLower"], conf["greenUpper"], 
                            conf["holeLower"], conf["holeUpper"])
        green_contours = frameHandler.detectGreen()

        # creating an instance of the BallDetector class to detect balls in the frame
        ballDetector = BallDetector(green_contours)
        
        # if the number of sides of the green area is 4 then the detected area is a rectangle
        # then we perform 4 point perspective transform to get the bird's eye view of 
        # the green
        if len(green_contours) == 4:
            # calculating the projection matrix for the green
            maxDims, M, _ = frameHandler.fourPointTransform(green_contours)

            # warping the frame to bird's eye view
            warp = cv2.warpPerspective(frame, M, maxDims)
            warp_height, warp_width = warp.shape[:2]

            # variable initialization related to calculation of speed, angle and make
            height_ratio = warp_height/conf["put_height"]; width_ratio = warp_width/conf["put_width"]
            velocity_start_pixel = conf["velocity_start"] * height_ratio
            velocity_stop_pixel = conf["velocity_stop"] * height_ratio; ballAngle = None; ballVelocity = None; ballMake = False
            
            # the hole co-ordinates in actual world
            holeDims = (conf["hole_x"], conf["hole_y"], conf["hole_w"], conf["hole_h"])
            holeCentres = (conf["hole_cX"], conf["hole_cY"])
            
            # finding the original and warped co-ordinates of the hole centres
            holerect, (hole_cX, hole_cY), (hole_cX_warped, hole_cY_warped) = frameHandler.detectHole(green_contours, height_ratio, 
                                                                                    width_ratio, holeDims, holeCentres)
            (hole_x, hole_y, hole_w, hole_h) = holerect
            
            base_line = np.array([hole_cX_warped, 0], dtype="float32") - np.array([hole_cX_warped, hole_cY_warped], dtype="float32")
            
            # creating an object for handling the dynamics of the ball
            balldyanmics = BallDynamics(velocity_start_pixel, velocity_stop_pixel, conf["fps"], holerect, (hole_cX, hole_cY))
            
            # creating a video writer
            vidout = cv2.VideoWriter(os.path.join(processed_output_path, video_name), fourcc, 30, 
                        (frame.shape[1], frame.shape[0]), True)
            vidout2 = cv2.VideoWriter(os.path.join(warped_output_path, video_name), fourcc, 30, 
                        (warp.shape[1], warp.shape[0]), True)

            while True:
                # using the instance of the BallDetector class to detect the ball
                ball_centers = ballDetector.detectBall(frame)
                minBallHoleDistance = float("inf")
                actualBallCenter = tuple()
                if ball_centers is not None:
                    if len(ball_centers) == 1:
                        actualBallCenter = ball_centers[0]
                    elif len(ball_centers) > 1:
                        for ball_center in ball_centers:
                            ballHoleDistance = distance.euclidean(ball_center, (hole_cX, hole_cY))
                            if ballHoleDistance < minBallHoleDistance:
                                minBallHoleDistance = ballHoleDistance
                                actualBallCenter = ball_center
                    cv2.circle(frame, actualBallCenter, 4, (0, 0, 255), -1)
                    ball_cX_warped, ball_cY_warped = warpPerspectiveCustom(actualBallCenter, M)
                    
                    # calculating the velocity of the ball in pixels/second
                    ballPixelVelocity = balldyanmics.velocityCal(ball_cX_warped, ball_cY_warped)
                    if ballPixelVelocity is not None:
                        # converting from pixels/second to metres/second
                        ballVelocity = (ballPixelVelocity/float(height_ratio))/1000.0
                        ballAngle = balldyanmics.angleCalc()
                    else:
                        ballVelocity = None
                    
                    # checking the make of the ball
                    ballMake = balldyanmics.ballMake(actualBallCenter[0], actualBallCenter[1])
                
                # drawing the contours on the frame
                cv2.drawContours(frame, [green_contours], -1, (0, 0, 255), 2)
                cv2.rectangle(frame, (hole_x, hole_y), (hole_x+hole_w, hole_y+hole_h), (255, 0, 0), 2)
                cv2.circle(frame, (hole_cX, hole_cY), 7, (255, 0, 0), -1)

                warp = cv2.warpPerspective(frame, M, maxDims)

                # writing the processed and warped videos
                vidout.write(frame)
                vidout2.write(warp)

                success, frame = vidcap.read()
                if success == False:
                    break
                frame = cv2.resize(frame, (int(frame.shape[1]/1.25), int(frame.shape[0]/1.25)))
            
            if ballAngle is not None and ballVelocity is not None:
                csv.csvUpdate(PID=str(PID), SID=str(SID), PtID=str(PtID), speed=round(ballVelocity,2), angle=round(ballAngle,2), 
                make=ballMake, processing=True)
            else:
                csv.csvUpdate(PID=str(PID), SID=str(SID), PtID=str(PtID), processing=False)
        
        else:
            csv.csvUpdate(PID=str(PID), SID=str(SID), PtID=str(PtID), processing=False)
    
    bar.finish()
os.system("spd-say 'Processing Done'")