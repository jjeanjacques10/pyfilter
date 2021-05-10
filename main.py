#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import os.path
import math

img = []
filter_type = ['No filter', 'blur faces', 'blur eyes', 'contour', 'gray']
filter_index = 0


def getFace(frame):
    path_face = "cascade\haarcascade_frontalface_default.xml"

    # Initializes the cascade classifier
    face_classifier = cv2.CascadeClassifier(path_face)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_classifier.detectMultiScale(
        img_rgb, scaleFactor=1.2, minNeighbors=5)

    return faces


def getEyes(frame):
    path_eye = "cascade\haarcascade_eye.xml"

    # Initializes the cascade classifier
    eye_classifier = cv2.CascadeClassifier(path_eye)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    eyes = eye_classifier.detectMultiScale(
        img_rgb, scaleFactor=1.2, minNeighbors=5)

    return eyes


def setGray(image):
    img_copy = image.copy()
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    return img_gray


def setContour(image):
    img_copy = image.copy()
    img_gray = setGray(image)
    edged = cv2.Canny(img_gray, 50, 150)
    dilate = cv2.dilate(edged, None, iterations=1)
    erode = cv2.erode(dilate, None, iterations=1)
    return erode


def setBlur(image, faces):
    img_copy = image.copy()
    if(len(faces) > 0):
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            ROI = img_copy[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(ROI, (71, 71), 0)
            img_copy[y:y+h, x:x+w] = blur
    return img_copy


def main():
    global img, filter_type, filter_index

    cv2.namedWindow("Mask Preview")
    # sets the video input for webcam
    video = cv2.VideoCapture(1)

    # config windows size
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if video.isOpened():  # try to get the first frame
        rval, frame = video.read()
    else:
        rval = False

    while rval:
        # passes the frame to the function and receives treated image in img
        key = 0
        try:
            cv2.setMouseCallback('Mask Preview', mouse_click)

            faces = getFace(frame)
            eyes = getEyes(frame)

            filter_selected = filter_type[filter_index]

            if(filter_selected == 'contour'):
                img = setContour(frame)
            elif(filter_selected == 'gray'):
                img = setGray(frame)
            elif(len(faces) > 0 and filter_selected == 'blur faces'):
                img = setBlur(frame, faces)
            elif(len(eyes) > 0 and filter_selected == 'blur eyes'):
                img = setBlur(frame, eyes)
            else:
                img = frame.copy()

            cv2.putText(img, filter_selected, (50, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Mask Preview", img)
            cv2.imshow("Original", frame)

            rval, frame = video.read()
            key = cv2.waitKey(20)
        except Exception as e:
            print(f"Error - {e}")
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("Mask Preview")
    cv2.destroyWindow("Original")
    video.release()


def mouse_click(event, x, y, flags, param):
    global img, filter_type, filter_index

    if event == cv2.EVENT_RBUTTONDOWN:
        filter_index = 0

    if event == cv2.EVENT_LBUTTONDOWN:
        if(filter_index < (len(filter_type) - 1)):
            filter_index += 1
        else:
            filter_index = 1


if __name__ == "__main__":
    main()
