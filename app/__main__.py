from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import numpy as np
import cv2
import threading
import queue
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
            filtered_corners, filtered_ids = detector.filter_markers_by_area(
                corners, ids, min_area=500, max_area=2000
            )

            ids = filtered_ids
            corners = filtered_corners
            
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

                # Perform web search with the decoded message
                searcher = WebSearch()
                searcher.search(message)
                
                # Show visualization windows
                cv2.imshow("Warped Perspective", warped)
                cv2.imshow("Character Recognition Results", result_with_matches)
                
                print("\nPress any key to close windows...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
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

