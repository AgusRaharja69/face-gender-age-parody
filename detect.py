import cv2
import time
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 0, 255), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(0)
padding = 20

# Define the initial position and size of the rectangle
rect_x, rect_y = 0, 0
rect_width = 150
rect_height = 200
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_x = frame_width // 2 - rect_width // 2
target_y = frame_height // 2 - rect_height // 2
increment_x = 4
increment_y = 4

# start_time = time.time()
move_rectangle = False
image_pop_up = False
image_pop_up2nd = False

# Load the image to pop up
pop_up_image = cv2.imread('penmaru_gel_2.jpeg')

if pop_up_image is None:
    print("Error: Image not found or could not be loaded. Check the path and image file.")
else:
    print("Image loaded successfully.")

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # current_time = time.time()
    # if current_time - start_time >= 10:
    #     move_rectangle = True

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
            max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
        ]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, age: {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if move_rectangle:
        # Draw and move the rectangle towards the target position
        cv2.rectangle(resultImg, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 3)
        cv2.putText(resultImg, 'Male, age: (100-120)', (rect_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        if rect_x < target_x:
            rect_x += increment_x
        if rect_y < target_y:
            rect_y += increment_y
        rect_x = min(rect_x, target_x)
        rect_y = min(rect_y, target_y)

        if rect_x == target_x and rect_y == target_y:
            image_pop_up = True

    if image_pop_up:
        multiple = 400
        for i in range(400):
            resized_image = cv2.resize(pop_up_image, (int(frame_width/(multiple-i)), int(frame_height/(multiple-i))))
            resultImg = resized_image

    cv2.imshow("Detecting age and gender", resultImg)

    key = cv2.waitKey(1)
    if key == ord('m'):
        start_time = time.time()
        move_rectangle = True

    if key == ord('q'):
        break

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()
