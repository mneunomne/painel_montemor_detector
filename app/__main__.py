import cv2
from aruco_detector import ArucoDetector

def main():
    # Initialize detector
    detector = ArucoDetector('./tests/azulejo7.png')
    
    # Detect markers
    print("Detecting ArUco markers...")
    corners, ids, result_img = detector.detect_markers()
    
    if ids is not None:
        smallest_id = min(ids.flatten())
        print(f"Smallest marker ID: {smallest_id}")
        
        # Get perspective transform
        matrix, size = detector.get_perspective_transform(corners, ids)
        
        if matrix is not None:
            # Apply perspective correction
            warped = cv2.warpPerspective(result_img, matrix, size)
            
            # Load templates and process grid
            templates = detector.load_templates()
            grid_results, result_with_matches, message = detector.process_grid(
                warped, templates, data_length=smallest_id, export_cells=True
            )
            
            # Display results
            print(f"\nFinal decoded message: '{message}'")
            
            cv2.imshow("Original with Markers", result_img)
            cv2.imshow("Warped Perspective", warped)
            cv2.imshow("Character Recognition Results", result_with_matches)
            
            print("\nPress any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not establish perspective transform")
    else:
        print("No markers detected!")


if __name__ == "__main__":
    main()