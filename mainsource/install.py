import cv2
from ultralytics import YOLO
import datetime

# Constants
YELLOW_LINE_HEIGHT_RATIO = 0.75
PEPSI_CAN_CLASS = 0
COCA_CAN_CLASS = 0
SEVENUP_CAN_CLASS = 0
FONT_SCALE = 0.5
FONT_THICKNESS = 2
LINE_THICKNESS = 2
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255,0,0)

# Global variables
session_start_time = None
session_end_time = None
session_number = 0
session_started = False

def load_models(pepsi_model_path: str, coca_model_path: str, sevenup_model_path: str) -> tuple:
    # Load models
    try:
        pepsi_model = YOLO(pepsi_model_path)
        coca_model = YOLO(coca_model_path)
        sevenup_model = YOLO(sevenup_model_path)
        return pepsi_model, coca_model, sevenup_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def draw_yellow_line(frame: cv2.Mat) -> None:
    # Draw a yellow line at 3/4 of the screen height
    line_x = int(frame.shape[1] * YELLOW_LINE_HEIGHT_RATIO)
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), YELLOW_COLOR, LINE_THICKNESS)

def detect_cans(frame: cv2.Mat, models: tuple) -> tuple:
    # Detect cans in the frame
    pepsi_model, coca_model, sevenup_model = models
    pepsi_results = pepsi_model.predict(source=frame, verbose=False)
    coca_results = coca_model.predict(source=frame, verbose=False)
    sevenup_results = sevenup_model.predict(source=frame, verbose=False)
    return pepsi_results, coca_results, sevenup_results

def draw_bounding_box(frame: cv2.Mat, bounding_box: tuple, color: tuple, confidence: float) -> None:
    # Draw bounding box around the detected object and display the confidence score
    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)
    confidence = float(confidence)
    cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

def check_intersection(bounding_box: tuple, line_x: int) -> bool:
    # Check if the bounding box intersects with the yellow line
    x1, _, x2, _ = bounding_box
    return x1 < line_x and x2 > line_x

def count_cans_crossing_yellow_line(frame: cv2.Mat, models: tuple) -> tuple:
    # Count cans crossing the yellow line
    pepsi_model, coca_model, sevenup_model = models
    pepsi_results, coca_results, sevenup_results = detect_cans(frame, models)
    pepsi_counter = 0
    coca_counter = 0
    sevenup_counter = 0
    for result in pepsi_results:
        for bounding_box in result.boxes:
            if bounding_box.cls == PEPSI_CAN_CLASS:
                box = [int(coord) for coord in bounding_box.xyxy.tolist()[0]]
                confidence = bounding_box.conf
                draw_bounding_box(frame, box, BLUE_COLOR, confidence)
                line_x = int(frame.shape[1] * YELLOW_LINE_HEIGHT_RATIO)
                if check_intersection(box, line_x):
                    pepsi_counter += 1
    for result in coca_results:
        for bounding_box in result.boxes:
            if bounding_box.cls == COCA_CAN_CLASS:
                box = [int(coord) for coord in bounding_box.xyxy.tolist()[0]]
                confidence = bounding_box.conf
                draw_bounding_box(frame, box, RED_COLOR, confidence)
                line_x = int(frame.shape[1] * YELLOW_LINE_HEIGHT_RATIO)
                if check_intersection(box, line_x):
                    coca_counter += 1
    for result in sevenup_results:
        for bounding_box in result.boxes:
            if bounding_box.cls == SEVENUP_CAN_CLASS:
                box = [int(coord) for coord in bounding_box.xyxy.tolist()[0]]
                confidence = bounding_box.conf
                draw_bounding_box(frame, box, GREEN_COLOR, confidence)
                line_x = int(frame.shape[1] * YELLOW_LINE_HEIGHT_RATIO)
                if check_intersection(box, line_x):
                    sevenup_counter += 1
    return pepsi_counter, coca_counter, sevenup_counter

def display_frame(frame: cv2.Mat, pepsi_counter: int, coca_counter: int, sevenup_counter: int) -> None:
    # Display the frame with counters
    text_position = (10, 30)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_type = cv2.LINE_AA

    # Draw counters
    cv2.putText(frame, f"Pepsi: {pepsi_counter}", (text_position[0], text_position[1] + 20), font, font_scale, text_color, font_thickness, line_type)
    cv2.putText(frame, f"Coca-Cola: {coca_counter}", (text_position[0], text_position[1] + 40), font, font_scale, text_color, font_thickness, line_type)
    cv2.putText(frame, f"7Up: {sevenup_counter}", text_position, font, font_scale, text_color, font_thickness, line_type)

    # Draw "Press 'c' to start session" text
    cv2.putText(frame, "Press 'c' to start session", (10, frame.shape[0] - 30), font, font_scale, text_color, font_thickness, line_type)
    if session_started:
        cv2.putText(frame, "Working...", (frame.shape[1] - 140, frame.shape[0] -20), font, font_scale, text_color, font_thickness, line_type)

    # Draw "Press 'e' to end" text
    cv2.putText(frame, "Press 'e' to end session", (10, frame.shape[0] - 10), font, font_scale, text_color, font_thickness, line_type)

    # Display the frame
    cv2.imshow("Can Counter", frame)


def start_counting_session():
    # Start a counting session
    global session_start_time, session_number, session_started
    session_start_time = datetime.datetime.now()
    session_number += 1
    session_started = True

def end_counting_session(pepsi_counter: int, coca_counter: int, sevenup_counter: int):
    # Export counted information into a text file
    global session_end_time, session_started
    session_end_time = datetime.datetime.now()
    filename = f"Session_{session_number}_{session_start_time}_{session_end_time}.txt"
    with open(filename, "w") as file:
        file.write("Session Information:\n")
        file.write(f">Start Time: {session_start_time}\n")
        file.write(f"<End Time  : {session_end_time}\n")
        file.write(f"Pepsi: {pepsi_counter}\n")
        file.write(f"Coca-Cola: {coca_counter}\n")
        file.write(f"7Up: {sevenup_counter}\n")
    print(f"Information exported to {filename}")
    session_started = False

def main():
    global session_number, session_started

    # Load models
    pepsi_model, coca_model, sevenup_model = load_models("pepsi_best.pt", "coca_best.pt", "7up_best.pt")
    if pepsi_model is None or coca_model is None or sevenup_model is None:
        return

    pepsi_counter = 0
    coca_counter = 0
    sevenup_counter = 0
    cooldown_counter = 0
    COOLDOWN = 20

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            draw_yellow_line(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and not session_started:
                start_counting_session()
                # Reset all counters
                pepsi_counter = 0
                coca_counter = 0
                sevenup_counter = 0
            elif key == ord("e") and session_started:
                end_counting_session(pepsi_counter, coca_counter, sevenup_counter)

            if cooldown_counter == 0:
                pepsi_count, coca_count, sevenup_count = count_cans_crossing_yellow_line(frame, (pepsi_model, coca_model, sevenup_model))
                pepsi_counter += pepsi_count
                coca_counter += coca_count
                sevenup_counter += sevenup_count
                if pepsi_count > 0 or coca_count > 0 or sevenup_count > 0:
                    cooldown_counter = COOLDOWN

            if cooldown_counter > 0:
                cooldown_counter -= 1

            display_frame(frame, pepsi_counter, coca_counter, sevenup_counter)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
