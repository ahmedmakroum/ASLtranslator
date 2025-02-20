import cv2
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    # Specify the model path where the trained model is saved
    model_path = "/home/arx/Projects/SignTrans/models/my_model.h5"
    recognizer = GestureRecognizer(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame, landmarks = tracker.detect_hand(frame)

        if landmarks:
            try:
                # Use the recognize method
                predicted_letter = recognizer.recognize(landmarks)  
            except AttributeError:
                print("Error: Method name might be incorrect. Check recognizer class.")
                predicted_letter = None
        else:
            predicted_letter = None  # No hand detected

        if predicted_letter:
            cv2.putText(frame, predicted_letter, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
