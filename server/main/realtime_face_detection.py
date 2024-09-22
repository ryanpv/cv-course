import cv2
import face_recognition

#
webcam_video_stream = cv2.VideoCapture(0)  # 0 for default, only one camera is connected

# hold all face locations in frame
all_face_locations = []

while True:
    # get current frame from video stream as img
    ret, current_frame = webcam_video_stream.read()

    # resize current frame to 25% to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # detect all faces in image
    all_face_locations = face_recognition.face.locations(
        current_frame_small, model="hog"
    )

    for index, current_face_location in enumerate(all_face_locations):
        # get four position values of current face
        top, right, bottom, left = current_face_location
        # change position magnitude to fit the actual size video frame
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4

        print(
            "Found face {} at top: {}, right{}, bottom: {}, left: {}".format(
                index + 1, top, right, bottom, left
            )
        )

        # draw rectangle around detected face
        cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # display current face with rectangle drawn
        cv2.imshow("Webcam video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the stream and cam
webcam_video_stream.release()
# close all opencv windows that are open
cv2.destroyAllWindows()
