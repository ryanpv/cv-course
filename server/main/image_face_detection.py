import cv2
import face_recognition

# image_to_detect = cv2.imread('images/ryan-profile.png')
image_to_detect = cv2.imread('images/group-photo.jpg')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')
# all_face_locations = face_recognition.face_locations(image_to_detect, model='cnn')

print("Number of faces in this photo is: {}".format(len(all_face_locations)))


# Looping through the face locations
for index, current_face_location in enumerate(all_face_locations):
    top, right, bottom, left = current_face_location
    print(
        "Found face {} at top: {}, right: {}, bottom: {}, left: {}".format(
            index + 1, top, right, bottom, left
        )
    )
    current_face_image = image_to_detect[top:bottom, left:right]
    cv2.imshow("Face no. " + str(index + 1), current_face_image)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
