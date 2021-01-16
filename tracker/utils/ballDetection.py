# import the necessary packages
from tracker.utils.custom import findCentre
from scipy.spatial import distance
import numpy as np
import cv2

class BallDetector():
    # This class handles the detection of ball in every frame
    def __init__(self, contours):
        self.contours = contours
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    def detectBall(self, frame):
        # if multiple ball_centers detected
        ball_centers = []
        # masking out the green from the complete frame
        green_mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.drawContours(green_mask, [self.contours], -1, 255, -1)
        green = cv2.bitwise_and(frame, frame, mask=green_mask)
        
        # to avoid detection of the ball and the golf stick in the beginning
        # masking out the ball and the stick with black
        pts = self.contours.reshape(4, 2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        topWidth = distance.euclidean(tr, tl)
        leftHeight = distance.euclidean(tr, br)
        
        # performing initial image processing to detect the ball using background subtraction
        green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        green_gray = cv2.GaussianBlur(green_gray, (5,5), 0)
        fgmask = self.fgbg.apply(green_gray)
        fgmask = cv2.erode(fgmask, None, iterations=3)
        fgmask[int(tl[1]):int(tl[1])+int(leftHeight/4), int(tl[0]):int(tl[0])+int(topWidth)] = 0
        
        # using the contours to detect the ball
        ball_contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # if ball contours present shortlisting the ball from the contours
        if len(ball_contours) > 0:
            for ball_c in ball_contours:
                ball_hull = cv2.convexHull(ball_c)
                (ball_x, ball_y, ball_w, ball_h) = cv2.boundingRect(ball_c)
                aspect_ratio = float(ball_w)/ball_h
                if 150 <= cv2.contourArea(ball_hull) <= 900 and 0.5 <= aspect_ratio <= 1.5:
                    ball_center = findCentre(ball_c)
                    ball_centers.append(ball_center)
        if len(ball_centers) > 0:
            return ball_centers
        else:
            return None