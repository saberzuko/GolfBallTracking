# import the necessary packages
from tracker.utils.custom import warpPerspectiveCustom
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
        self.rect = None
        self.dist = None
    
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
            self.rect = np.zeros((4,2), dtype="float32")
            s = approx.sum(axis=1)
            self.rect[0] = approx[np.argmin(s)]
            self.rect[2] = approx[np.argmax(s)]
            diff = np.diff(approx, axis=1)
            self.rect[1] = approx[np.argmin(diff)]
            self.rect[3] = approx[np.argmax(diff)]
            (tl, tr, br, bl) = self.rect
            br[1] += 10
            bl[1] += 10
            approx = np.array([tr, tl, bl, br], dtype=np.int32)
        return approx
    
    def detectHole(self, contours, height_ratio, width_ratio, holeDims, holeCentres):
        
        # calculating the hole co-ordinates in the wapred figure from the real world co-ordinates
        (_, warpHeight), _, MInverse = self.fourPointTransform(contours)
        hole_cX_warped = int(holeCentres[0] * width_ratio)
        hole_cY_warped = int(warpHeight - (holeCentres[1] * height_ratio))
        hole_x_warped = int(holeDims[0] * width_ratio)
        hole_y_warped = int(warpHeight - (holeDims[1] * height_ratio)) - 5
        hole_w_warped = int(holeDims[2] * width_ratio)
        hole_h_warped = int(holeDims[3] * height_ratio) - 5

        # using the inverse perspective transform to get the hole co-ordinates in the original video
        hole_cX, hole_cY = warpPerspectiveCustom((hole_cX_warped, hole_cY_warped), MInverse)
        hole_x, hole_y = warpPerspectiveCustom((hole_x_warped, hole_y_warped), MInverse)
        hole_xw, hole_yh = warpPerspectiveCustom((hole_x_warped + hole_w_warped, hole_y_warped + hole_h_warped), MInverse)
        hole_w = hole_xw - hole_x
        hole_h = hole_yh - hole_y
        holerect = (hole_x, hole_y, hole_w, hole_h)
        hole_centre = (hole_cX, hole_cY)
        hole_centre_warped = (hole_cX_warped, hole_cY_warped)
        
        return(holerect, hole_centre, hole_centre_warped)

    
    def fourPointTransform(self, contours):
        # defining the four point transform to get the bird's eye view of the green area
        pts = contours.reshape(4, 2)
        self.rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis = 1)
        
        # the minimum sum of the co-ordinates (x+y) correspond to the top left corner of the green
        self.rect[0] = pts[np.argmin(s)]
        
        # the maximum sum of the co-ordinates correspond to the bottom right of the green
        self.rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        
        # the minimum difference of the co-ordinates (x-y) corresponds to the top right corner of the green
        self.rect[1] = pts[np.argmin(diff)]
        
        # the maximum difference of the co-ordinates (x-y) corresponds to the bottom left corner of the green
        self.rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = self.rect
        
        # computing the width and the height of each side
        bottomWidth = distance.euclidean(br, bl)
        topWidth = distance.euclidean(tr, tl)
        leftHeight = distance.euclidean(tr, br)
        rightHeight = distance.euclidean(tl, bl)

        maxWidth = int(max(bottomWidth, topWidth))
        maxHeight = int(max(leftHeight, rightHeight))
        
        # new co-ordinates of the frame from the bird's eye view.
        self.dist = np.array([[0,0], [int(maxWidth)-1, 0], [int(maxWidth)-1, int(maxHeight)-1], [0, int(maxHeight)-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(self.rect, self.dist)
        M_inverse = cv2.getPerspectiveTransform(self.dist, self.rect)
        return ((maxWidth, maxHeight), M, M_inverse)