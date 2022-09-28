import cv2

def video_detection():

    video = cv2.VideoCapture('carv.mp4')

    car_tracker_file = 'carx.xml'
    pedestrian_tracker_file = 'full_body_detection.xml'

    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

    car_tracker = cv2.CascadeClassifier(car_tracker_file)

    while True:
        (read_successful, frame) = video.read()

        if read_successful:
            grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        cars = car_tracker.detectMultiScale(grey_img)
        pedestrians = pedestrian_tracker.detectMultiScale(grey_img)

        print(cars)

        for(x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('car_detector', frame)

        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break


video_detection()
print("Works Fine!")