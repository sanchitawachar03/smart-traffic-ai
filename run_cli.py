import cv2
import sys
from detector import TrafficDetector

def run_traffic_system(video_path=None):
    detector = TrafficDetector('yolov8n.pt')
    
    # Open video or default webcam (0)
    if video_path is None or video_path == '':
        print("[INFO] No video provided. Enter video path (e.g. sample.mp4) or press Enter for webcam: ")
        user_input = input().strip()
        video_source = int(user_input) if user_input.isdigit() else (user_input if user_input else 0)
    else:
        video_source = video_path

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return

    print("[INFO] Traffic Detection Started! Press 'q' in video window to exit.")
    print("[INFO] Press 'e' key to toggle EMERGENCY AMBULANCE override.")

    emergency_toggle = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video reached.")
            break

        # Resize for smooth display
        frame = cv2.resize(frame, (960, 540))

        # AI Detection
        annotated_frame, counts, total_vehicles, density_score, is_emergency = detector.process_frame(
            frame, conf_threshold=0.35, emergency_override=emergency_toggle
        )

        # Dynamic Signal Decision
        decision = detector.calculate_signal_decision(density_score, total_vehicles, is_emergency)

        # Draw HUD / Dashboard on frame
        cv2.rectangle(annotated_frame, (10, 10), (450, 140), (20, 20, 20), -1)
        cv2.rectangle(annotated_frame, (10, 10), (450, 140), (0, 255, 255), 2)

        # Signal status text
        signal_color = (0, 255, 0) if decision['light_state'] == 'GREEN' else (0, 0, 255)
        cv2.putText(annotated_frame, f"SIGNAL: {decision['signal']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, signal_color, 2)
        cv2.putText(annotated_frame, f"Green Time: {decision['allocated_time']}s | Score: {density_score:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Cars: {counts['Car']} | Bikes: {counts['Motorcycle']} | Bus: {counts['Bus']} | Truck: {counts['Truck']}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"Traffic Level: {decision['level']}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("AI Smart Traffic Signal Controller", annotated_frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            emergency_toggle = not emergency_toggle
            print(f"[EMERGENCY OVERRIDE]: {'ACTIVATED' if emergency_toggle else 'DEACTIVATED'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_traffic_system(video_arg)
