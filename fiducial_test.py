import cv2
import numpy as np

def detect_aruco_on_red_clay(image_path=None, camera_index=0, use_camera=True):
    """
    Detect ArUco markers on red clay surface with optimized parameters
    Clay color: RGB(180, 80, 53)
    """
    
    # Initialize ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimized parameters for red clay surface detection
    # Adaptive thresholding parameters
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    
    # Contour filtering parameters
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    
    # Bits extraction parameters
    parameters.markerBorderBits = 1
    parameters.minOtsuStdDev = 5.0
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # Corner refinement for better accuracy with blur
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1
    
    # Error correction
    parameters.errorCorrectionRate = 0.6
    
    def preprocess_image(img):
        """Preprocess image for better marker detection on red clay"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract red channel (better contrast on red clay)
        red_channel = img[:, :, 2]  # BGR format, so red is index 2
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(red_channel, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return filtered
    
    def detect_markers(img):
        """Detect ArUco markers in the image"""
        # Preprocess the image
        processed = preprocess_image(img)
        
        # Convert back to 3-channel for ArUco detection
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed_bgr, aruco_dict, parameters=parameters
        )
        
        return corners, ids, rejected, processed
    
    if use_camera:
        # Use camera feed
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect markers
            corners, ids, rejected, processed = detect_markers(frame)
            
            # Draw detected markers
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                print(f"Detected markers: {ids.flatten()}")
            
            # Show original and processed images
            cv2.imshow('Original', frame)
            cv2.imshow('Processed (Red Channel)', processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('aruco_detection_frame.jpg', frame)
                cv2.imwrite('processed_frame.jpg', processed)
                print("Frames saved!")
        
        cap.release()
    
    else:
        # Use static image
        if image_path is None:
            print("Please provide image_path when use_camera=False")
            return
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Detect markers
        corners, ids, rejected, processed = detect_markers(img)
        
        # Draw results
        result_img = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
            print(f"Detected markers: {ids.flatten()}")
            
            # Print corner coordinates
            for i, corner_set in enumerate(corners):
                marker_id = ids[i][0]
                print(f"Marker {marker_id} corners:")
                for j, corner in enumerate(corner_set[0]):
                    print(f"  Corner {j}: ({corner[0]:.2f}, {corner[1]:.2f})")
        else:
            print("No markers detected")
        
        # Display results
        cv2.imshow('Original Image', img)
        cv2.imshow('Detection Result', result_img)
        cv2.imshow('Processed (Red Channel)', processed)
        
        print("Press any key to close windows")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def create_test_marker(marker_id=0, size=200):
    """Create a test ArUco marker for testing"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add white border for better detection
    bordered = cv2.copyMakeBorder(marker_img, 50, 50, 50, 50, 
                                  cv2.BORDER_CONSTANT, value=255)
    
    cv2.imwrite(f'test_marker_{marker_id}.png', bordered)
    print(f"Test marker {marker_id} saved as 'test_marker_{marker_id}.png'")
    
    return bordered

if __name__ == "__main__":
    # Create a test marker first
    create_test_marker(0)
    create_test_marker(1)
    
    # Load and process image.jpg
    print("Loading image.jpg for ArUco detection...")
    detect_aruco_on_red_clay(image_path='azulejo3.jpg', use_camera=False)