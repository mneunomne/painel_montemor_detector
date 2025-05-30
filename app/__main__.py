import cv2
from aruco_detector import ArucoDetector
from web_search import WebSearch


def main():
    # Initialize detector
    detector = ArucoDetector('http://192.168.31.227:5000/video_feed')
    
    # Detect markers with accumulation over multiple frames
    print("Detecting ArUco markers with frame accumulation...")
    corners, ids, result_img = detector.detect_markers_with_accumulation(
        max_attempts=10,  # Try up to 100 frames
    )
    
    if corners is not None and ids is not None:
        # draw src_pts on the result image with polyline
        
        cv2.imshow("Original with Accumulated Markers", result_img)
        
        # Find the smallest marker ID (used for data length)
        smallest_id = min(ids.flatten())
        print(f"Smallest marker ID: {smallest_id}")

        if (len(ids) == 4):
            
            # Get perspective transform
            matrix, size, src_pts = detector.get_perspective_transform(corners, ids, smallest_id)
            
            # draw src_pts as polyline
            cv2.polylines(result_img, [src_pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)



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
        else:
            print("Could not establish perspective transform")
            
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    else:
        print("No markers detected after all attempts!")
    


if __name__ == "__main__":
    main()