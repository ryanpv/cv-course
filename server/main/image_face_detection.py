import cv2
import face_recognition

# image_to_detect = cv2.imread('images/ryan-profile.png')
image_to_detect = cv2.imread('images/group-photo.jpg')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')
# all_face_locations = face_recognition.face_locations(image_to_detect, model='cnn')

print("Number of faces in this photo is: {}".format(len(all_face_locations)))