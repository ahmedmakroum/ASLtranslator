import cv2
import mediapipe as mp
import random

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                self.display_text(frame)

        return frame

    def display_text(self, frame):
        text = random.choice(["OK", "Hello"])
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.detect_hand(frame)
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = tracker.detect_hand(frame)
        cv2.imshow("Hand Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
