from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import numpy as np
import cv2
import threading
import queue
import time
# Add your imports here - adjust based on your actual module names
from aruco_detector import ArucoDetector  # Adjust import path as needed
from web_search import WebSearch  # Adjust import path as needed

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='static')

FLASK_SERVER_IP="0.0.0.0"
FLASK_SERVER_PORT=3333

socketio = SocketIO(app)

video_output = None
cropped_output = None
avaraged_output = None
live_output = None
capture_avarage_frames = False

current_segment_index = 0

# Global queue for ArUco detection requests (like your detections_queue)
detection_request_queue = queue.Queue()


def find_longest_consecutive_sequence(corners, ids, max_markers=4):
    """Find the longest consecutive sequence of marker IDs"""
    if ids is None or len(ids) == 0:
        return corners, ids
    
    flat_ids = ids.flatten()
    unique_ids = sorted(np.unique(flat_ids))
    
    # Find all consecutive sequences
    sequences = []
    current_sequence = [unique_ids[0]]
    
    for i in range(1, len(unique_ids)):
        if unique_ids[i] == unique_ids[i-1] + 1:  # Consecutive
            current_sequence.append(unique_ids[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [unique_ids[i]]
    
    sequences.append(current_sequence)  # Add the last sequence
    
    # Find the longest sequence (up to max_markers)
    best_sequence = []
    for seq in sequences:
        if len(seq) > len(best_sequence):
            best_sequence = seq[:max_markers]  # Limit to max_markers
    
    print(f"Available sequences: {sequences}")
    print(f"Selected sequence: {best_sequence}")
    
    # Filter markers to keep only the best consecutive sequence
    filtered_corners = []
    filtered_ids = []
    
    for i, marker_id in enumerate(flat_ids):
        if marker_id in best_sequence:
            filtered_corners.append(corners[i])
            filtered_ids.append([marker_id])
    
    return filtered_corners, np.array(filtered_ids) if filtered_ids else None

def is_square_shaped(corner_points, tolerance=0.3):
    """Check if marker is roughly square-shaped"""
    # Calculate all 4 side lengths
    side_lengths = []
    for j in range(4):
        p1 = corner_points[j]
        p2 = corner_points[(j + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        side_lengths.append(side_length)
    
    # Check if all sides are roughly equal
    avg_side = np.mean(side_lengths)
    max_deviation = max(abs(side - avg_side) for side in side_lengths)
    
    # Allow some tolerance for perspective distortion
    return max_deviation / avg_side <= tolerance

def filter_square_markers(corners, ids, tolerance=0.3):
    """Filter out markers that are not square-shaped"""
    if ids is None or len(ids) == 0:
        return corners, ids
    
    filtered_corners = []
    filtered_ids = []
    
    for i, corner_set in enumerate(corners):
        corner_points = corner_set[0] if len(corner_set.shape) == 3 else corner_set
        
        if is_square_shaped(corner_points, tolerance):
            filtered_corners.append(corners[i])
            filtered_ids.append(ids[i])
        else:
            print(f"Marker ID {ids[i][0]} filtered out - not square enough")
    
    return filtered_corners, np.array(filtered_ids) if filtered_ids else None


def run_aruco_detection():
    """Run the ArUco detection process with OpenCV windows (main thread only)"""
    try:
        # Initialize detector
        detector = ArucoDetector('http://192.168.31.227:5000/video_feed')
        
        # Detect markers with accumulation over multiple frames
        print("Detecting ArUco markers with frame accumulation...")
        corners, ids, result_img = detector.detect_markers_with_accumulation(
            max_attempts=10,  # Try up to 10 frames
        )
        
        if corners is not None and ids is not None:
            # Process detected markers
            if corners is not None:
                for corner, index in zip(corners, range(len(corners))):
                    size = cv2.contourArea(corner[0])
                    id = ids[index][0] if ids is not None else -1

            print(f"ids: {ids}")
            print(f"corners: {corners}")

            # Filter corners and ids by size
            corners, ids = detector.filter_markers_by_area(
                corners, ids, min_area=500, max_area=2000
            )
            
            # Apply all filters in sequence
            corners, ids = detector.filter_markers_by_area(corners, ids, min_area=500, max_area=2000)
            if ids is not None:
                print(f"After size filter: {ids.flatten()}")

            corners, ids = filter_square_markers(corners, ids, tolerance=0.3)
            if ids is not None:
                print(f"After shape filter: {ids.flatten()}")

            corners, ids = find_longest_consecutive_sequence(corners, ids, max_markers=4)
            
            # Show original with markers
            cv2.imshow("Original with Accumulated Markers", result_img)
            
            # Find the smallest marker ID (used for data length)
            smallest_id = min(ids.flatten())
            print(f"Smallest marker ID: {smallest_id}")
                

            if (len(ids) == 4):
                # Get perspective transform
                matrix, size, src_pts = detector.get_perspective_transform(corners, ids, smallest_id)
                
                # Draw src_pts as polyline
                cv2.polylines(result_img, [src_pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Show updated image with polyline
                cv2.imshow("Original with Accumulated Markers", result_img)

                # Apply perspective correction
                warped = cv2.warpPerspective(result_img, matrix, size)
                
                # Load templates and process grid
                templates = detector.load_templates()
                grid_results, result_with_matches, message = detector.process_grid(
                    warped, templates, data_length=smallest_id, export_cells=True
                )
                
                # Display results
                print(f"\nFinal decoded message: '{message}'")

                # Show visualization windows
                cv2.imshow("Warped Perspective", warped)
                cv2.imshow("Character Recognition Results", result_with_matches)

                # cv wait sometime and let the window show
                cv2.waitKey(1000)
                # Close the windows after showing
                cv2.destroyAllWindows()

                # year is last 4 characters
                # search_message = message.replace('|', ' ')
                # last 4 characters of message
                message = message.replace('?', '')
                year = message[-4:]
                name = message[:-4].strip()

                year = year.replace('T', '1')
                year = year.replace('|', '1')
                year = year.replace('O', '0')
                year = year.replace('|', ' ')
                year = year.replace('Ç', '5')
                year = year.replace('B', '8')
                name = name.replace('1', 'T')
                name = name.replace('7', 'M')

                search_message = f"{name} ano {year}"

                # Perform web search with the decoded message
                searcher = WebSearch()
                searcher.search(search_message)
            
                
                #print("\nPress any key to close windows...")
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # replace "|" with " ""                
                return f"Successfully decoded message: '{message}'"
            else:
                print("Could not establish perspective transform - need exactly 4 markers")
                print("\nPress any key to close windows...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return "Could not establish perspective transform - need exactly 4 markers"
        else:
            print("No markers detected after all attempts!")
            return "No markers detected after all attempts!"
            
    except Exception as e:
        print(f"Error in ArUco detection: {e}")
        return f"Error: {str(e)}"

@app.route('/read_tile')
def read_tile():
    global ready_to_read
    ready_to_read = True

    # Run ArUco detection in a separate thread to avoid blocking the HTTP response
    def detection_thread():
        result = run_aruco_detection()
        print(f"ArUco detection result: {result}")
    
    detection_request_queue.put(1)
    
    return Response(f'done', mimetype='text/plain')

def main():
    """Main function following your working pattern"""
    global ready_to_read
    
    # Initialize variables
    ready_to_read = False
    
    print(f"Starting Flask server on {FLASK_SERVER_IP}:{FLASK_SERVER_PORT}")
    
    # Start Flask in a separate thread (exactly like your working code)
    thread_flask = threading.Thread(target=lambda: socketio.run(
        app, 
        host=FLASK_SERVER_IP, 
        port=FLASK_SERVER_PORT, 
        debug=False,  # Set to False when running in thread
        allow_unsafe_werkzeug=True
    ))
    thread_flask.daemon = True
    thread_flask.start()
    
    print("Flask server started in background thread")
    print("Main thread ready for OpenCV operations...")
    
    # Create OpenCV windows (like your code does)
    cv2.namedWindow("Main ArUco Detection")
    
    # Main loop similar to your run_opencv() method
    try:
        while True:
            # Check for detection requests (like your queue.get_nowait() pattern)
            try:
                direction = detection_request_queue.get_nowait()
                print(f"Processing ArUco detection for direction: {direction}")
                result = run_aruco_detection()
                print(f"ArUco detection result: {result}")
            except queue.Empty:
                # No detection request, continue loop
                pass
            
            # Keep the main thread alive and allow OpenCV to process events
            # This is similar to your cv2.waitKey(1) in the main loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

