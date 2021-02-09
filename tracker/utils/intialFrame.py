# import the necessary packages
from scipy.spatial import distance
from skimage import measure
import numpy as np
import cv2

# the class is defined to handle the initial frame and detect the green put
# and the hole
class FrameHandling():
    def __init__(self, frame, greenLower, greenUpper, holeLower, holeUpper):
        self.frame = frame
        # initializing the HSV values for the green and the hole
        # for detecting the green and detecting the hole
        self.greenLower = np.array(greenLower)
        self.greenUpper = np.array(greenUpper)
        self.holeLower = np.array(holeLower)
        self.holeUpper = np.array(holeUpper)
    
    def detectGreen(self):
        # performing initial image processing to detect the green area in the frame
        blurred = cv2.GaussianBlur(self.frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        green_thresh = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
        green_thresh = cv2.morphologyEx(green_thresh, cv2.MORPH_CLOSE, kernel)

        # performing connected component analysis to take the area with the highest green region
        labels = measure.label(green_thresh, connectivity=2, background=0)
        mask = np.zeros(green_thresh.shape, dtype=np.uint8)
        for (i,label) in enumerate(np.unique(labels)):
            if label == 0:
                continue
            labelMask = np.zeros(green_thresh.shape, dtype=np.uint8)
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 5000:
                mask = cv2.add(mask, labelMask)
        
        # detecting the contours of the green region and approximating it to a rectangle
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.0185*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if (len(approx) == 4):
            approx = np.reshape(approx, (4,2))
            rect = np.zeros((4,2), dtype="float32")
            s = approx.sum(axis=1)
            rect[0] = approx[np.argmin(s)]
            rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=1)
            rect[1] = approx[np.argmin(diff)]
            rect[3] = approx[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            br[1] += 10
            bl[1] += 10
            approx = np.array([tr, tl, bl, br], dtype=np.int32)
        return approx
    
    def detectHole(self, contours):
        # function to detect the hole in the green area
        green_mask = np.zeros(self.frame.shape[:2], dtype="uint8")
        
        # grabbing the green contours and using the contours to mask the green area
        cv2.drawContours(green_mask, [contours], -1, 255, -1)
        green = cv2.bitwise_and(self.frame, self.frame, mask=green_mask)
        green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
        hole_thresh = cv2.inRange(green_hsv, self.holeLower, self.holeUpper)
        
        # image processing to detect the hole
        hole_thresh = cv2.erode(hole_thresh, None, iterations=1)
        hole_thresh = cv2.dilate(hole_thresh, None, iterations=10)

        # detecting the contours of the hole region
        hole_contours, _ = cv2.findContours(hole_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hole_c = max(hole_contours, key=cv2.contourArea)
        return hole_c

    
    def fourPointTransform(self, contours):
        # defining the four point transform to get the bird's eye view of the green area
        pts = contours.reshape(4, 2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis = 1)
        
        # the minimum sum of the co-ordinates (x+y) correspond to the top left corner of the green
        rect[0] = pts[np.argmin(s)]
        
        # the maximum sum of the co-ordinates correspond to the bottom right of the green
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        
        # the minimum difference of the co-ordinates (x-y) corresponds to the top right corner of the green
        rect[1] = pts[np.argmin(diff)]
        
        # the maximum difference of the co-ordinates (x-y) corresponds to the bottom left corner of the green
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        
        # computing the width and the height of each side
        bottomWidth = distance.euclidean(br, bl)
        topWidth = distance.euclidean(tr, tl)
        leftHeight = distance.euclidean(tr, br)
        rightHeight = distance.euclidean(tl, bl)

        maxWidth = int(max(bottomWidth, topWidth))
        maxHeight = int(max(leftHeight, rightHeight))
        
        # new co-ordinates of the frame from the bird's eye view.
        dist = np.array([[0,0], [int(maxWidth)-1, 0], [int(maxWidth)-1, int(maxHeight)-1], [0, int(maxHeight)-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dist)
        return ((maxWidth, maxHeight), M)