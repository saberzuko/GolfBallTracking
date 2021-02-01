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
        
        # calculting the co-ordinates of the mask to hide the golf stick and masking the top quarter of the rectangle contour
        if (tr[0] < br[0]):
            mask_br_x = int(tr[0] + ((br[0] - tr[0])/2))
        else:
            mask_br_x = int(tr[0] - ((tr[0] - br[0])/2))
        mask_br_y = int(tr[1] + ((br[1] - tr[1])/4))
        mask_br = np.array([mask_br_x, mask_br_y], dtype=np.int32)
        if (tl[0] < bl[0]):
            mask_bl_x = int(tl[0] + ((bl[0] - tl[0])/2))
        else:
            mask_bl_x = int(tl[0] - ((tl[0] - bl[0])/2))
        mask_bl_y = int(tl[1] + ((bl[1] - tl[1])/4))
        mask_bl = np.array([mask_bl_x, mask_bl_y], dtype=np.int32)
        mask_contours = np.array([tr, tl, mask_bl, mask_br], dtype=np.int32)
        
        # performing initial image processing to detect the ball using background subtraction
        green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        green_gray = cv2.GaussianBlur(green_gray, (5,5), 0)
        fgmask = self.fgbg.apply(green_gray)
        cv2.drawContours(fgmask, [mask_contours], -1, 0, -1)
        fgmask = cv2.erode(fgmask, None, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=3)
        
        # using the contours to detect the ball
        ball_contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # if ball contours present shortlisting the ball from the contours
        if len(ball_contours) > 0:
            for ball_c in ball_contours:
                # ball_hull = cv2.convexHull(ball_c)
                # (ball_x, ball_y, ball_w, ball_h) = cv2.boundingRect(ball_c)
                # aspect_ratio = float(ball_w)/ball_h
                # if 150 <= cv2.contourArea(ball_hull) <= 900 and 0.5 <= aspect_ratio <= 1.5:
                try:
                    ball_center = findCentre(ball_c)
                    ball_centers.append(ball_center)
                except:
                    pass
        if len(ball_centers) > 0:
            return ball_centers
        else:
            return None