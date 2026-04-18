import cv2
from detector.model_loader import load_model
from detector.people_detector import detect_people
from detector.effects import add_anime_effect

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(model, frame)

        for box in detections:
            frame = add_anime_effect(frame, box)

        cv2.imshow("AnimeVision: AI People Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
