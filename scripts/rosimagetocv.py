#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from Lanedetect import LaneDetector
import numpy as np
Lane_Detector = LaneDetector()
def process_image(image,plot=False):
        mtx=np.loadtxt('/home/karthik/Downloads/opencv/camera_01/cameraMatrix.txt', delimiter=',', dtype=None)
        dist=np.loadtxt('/home/karthik/Downloads/opencv/camera_01/cameraDistortion.txt', delimiter=',', dtype=None)
        frame1=Lane_Detector.undistorth(image,mtx,dist)
        frame11=Lane_Detector.enhancement(frame1)
        lab=cv2.cvtColor(frame11, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        bitnotimage=cv2.bitwise_not(b)
        diff=bitnotimage-b
        diff[diff< 150] = 0
        multipyimage=np.multiply(diff,b)
        multipyimage[multipyimage<110] = 0
        DynamicI= Lane_Detector.dynamicrange(multipyimage)
        ROIimage=Lane_Detector.ROI(DynamicI)
        BIEimage,matrix=Lane_Detector.birdeyeview(ROIimage)
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds=Lane_Detector.extract_lanes_pixels(BIEimage)
        left_fit, right_fit, ploty, left_fitx, right_fitx=Lane_Detector.poly_fit( leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, BIEimage,plot)
        curvature=Lane_Detector.compute_curvature(left_fit, right_fit, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty)
        new_image=Lane_Detector.plain_lane(frame1, BIEimage, matrix, left_fitx, right_fitx, ploty, plot)
        offset_from_centre=Lane_Detector.compute_center_offset(curvature, new_image, plot)
        Final_image=Lane_Detector.render_curvature_and_offset(new_image, curvature, offset_from_centre, plot)
        return Final_image
class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/raspicam_node/image",Image,self.callback)

  

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    final=process_image(cv_image,True)
     

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    