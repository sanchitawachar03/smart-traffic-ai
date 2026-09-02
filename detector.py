import cv2
import numpy as np
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initializes the YOLO model for vehicle detection and ByteTrack tracking.
        """
        self.model_name = model_name
        self.model = YOLO(model_name)
        
        # COCO class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck'
        }
        
        # Space-occupancy weights for Indian / Urban traffic
        self.weights = {
            'Car': 2.0,
            'Motorcycle': 1.0,
            'Bus': 4.0,
            'Truck': 4.0,
            'Emergency': 6.0
        }

    def switch_model(self, new_model_name):
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self.model = YOLO(new_model_name)

    def process_frame(self, frame, conf_threshold=0.35, iou_threshold=0.5, emergency_override=False):
        """
        Runs YOLO object detection and tracking on a video frame.
        """
        # Run tracking (ByteTrack) to keep consistent vehicle IDs and eliminate flicker
        results = self.model.track(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        annotated_frame = frame.copy()
        counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0, 'Emergency': 0}
        tracked_ids = set()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None

                if cls_id in self.target_classes:
                    label_name = self.target_classes[cls_id]
                    counts[label_name] += 1
                    if track_id is not None:
                        tracked_ids.add(track_id)

                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    colors = {
                        'Car': (0, 230, 118),       # Neon Green
                        'Motorcycle': (255, 179, 0), # Amber
                        'Bus': (233, 30, 99),        # Magenta/Pink
                        'Truck': (0, 176, 255)       # Sky Blue
                    }
                    box_color = colors.get(label_name, (0, 255, 0))

                    # Draw stylish rounded corner box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                    # Label badge
                    id_tag = f" #{track_id}" if track_id else ""
                    label_text = f"{label_name}{id_tag} ({conf:.2f})"
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated_frame, (x1, max(y1 - 20, 0)), (x1 + w + 8, max(y1, 20)), box_color, -1)
                    cv2.putText(annotated_frame, label_text, (x1 + 4, max(y1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        is_emergency = emergency_override
        if is_emergency:
            counts['Emergency'] = 1
            # Emergency Banner
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 45), (0, 0, 220), -1)
            cv2.putText(annotated_frame, "EMERGENCY VEHICLE DETECTED - SIGNAL OVERRIDE", 
                        (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        total_vehicles = sum(counts.values())
        density_score = (
            counts['Car'] * self.weights['Car'] +
            counts['Motorcycle'] * self.weights['Motorcycle'] +
            counts['Bus'] * self.weights['Bus'] +
            counts['Truck'] * self.weights['Truck'] +
            counts['Emergency'] * self.weights['Emergency']
        )

        return annotated_frame, counts, total_vehicles, density_score, is_emergency

    def calculate_signal_timing(self, density_score, total_vehicles, is_emergency=False):
        """
        Calculates optimal green time and state based on real-time traffic volume.
        """
        if is_emergency:
            return {
                'recommended_green': 45,
                'level': 'EMERGENCY OVERRIDE',
                'status': '🚨 Instant Green Corridor granted to Emergency Vehicle!',
                'color_hex': '#00E676',
                'priority': 'URGENT'
            }

        if total_vehicles == 0 or density_score == 0:
            return {
                'recommended_green': 5,
                'level': 'Zero Traffic (Empty)',
                'status': '🛑 Road is clear. Short or Red signal to conserve cross-traffic time.',
                'color_hex': '#FF5252',
                'priority': 'IDLE'
            }

        if density_score <= 4:
            green_time = int(np.clip(10 + density_score * 2.0, 10, 18))
            return {
                'recommended_green': green_time,
                'level': 'Light Traffic',
                'status': f'🟢 Minimal congestion. Short {green_time}s green phase.',
                'color_hex': '#69F0AE',
                'priority': 'LOW'
            }
        elif density_score <= 12:
            green_time = int(np.clip(18 + (density_score - 4) * 2.2, 20, 35))
            return {
                'recommended_green': green_time,
                'level': 'Moderate Traffic',
                'status': f'🟢 Normal flow. Optimal {green_time}s green phase.',
                'color_hex': '#00E676',
                'priority': 'NORMAL'
            }
        else:
            green_time = int(np.clip(35 + (density_score - 12) * 1.5, 38, 60))
            return {
                'recommended_green': green_time,
                'level': 'Heavy Congestion',
                'status': f'🔥 High traffic backlog! Maximum {green_time}s green phase.',
                'color_hex': '#FFD600',
                'priority': 'HIGH'
            }