import cv2
import numpy as np

class InteractiveArucoDetector:
    def __init__(self, image_path='image.jpg'):
        """
        Interactive ArUco detector with real-time parameter adjustment
        """
        self.image_path = image_path
        self.original_image = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Initialize parameters with default values optimized for red clay
        self.params = {
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 23,
            'adaptiveThreshWinSizeStep': 10,
            'adaptiveThreshConstant': 0,
            'minMarkerPerimeterRate': 3,  # *100 for trackbar (0.03)
            'maxMarkerPerimeterRate': 400,  # *100 for trackbar (4.0)
            'polygonalApproxAccuracyRate': 3,  # *100 for trackbar (0.03)
            'minCornerDistanceRate': 5,  # *100 for trackbar (0.05)
            'minDistanceToBorder': 3,
            'markerBorderBits': 1,
            'minOtsuStdDev': 50,  # *10 for trackbar (5.0)
            'perspectiveRemovePixelPerCell': 8,
            'perspectiveRemoveIgnoredMarginPerCell': 13,  # *100 for trackbar (0.13)
            'cornerRefinementWinSize': 5,
            'cornerRefinementMaxIterations': 30,
            'cornerRefinementMinAccuracy': 10,  # *100 for trackbar (0.1)
            'errorCorrectionRate': 60,  # *100 for trackbar (0.6)
            'gaussianBlurKernel': 5,
            'claheClipLimit': 20,  # *10 for trackbar (2.0)
            'claheTileSize': 8,
            'bilateralD': 9,
            'bilateralSigmaColor': 75,
            'bilateralSigmaSpace': 75,
            'useRedChannel': 1,  # 0 = grayscale, 1 = red channel
            'contrastBrightness': 10,  # *10 for trackbar (1.0)
            'brightnessOffset': 0
        }
        
        self.setup_ui()
        self.load_image()
    
    def setup_ui(self):
        """Setup the interactive UI with trackbars"""
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Controls', 500, 900)
        cv2.moveWindow('Controls', 50, 50)  # Position the window explicitly
        
        # Create a dummy image for the controls window to ensure it's visible
        dummy = np.zeros((900, 500, 3), dtype=np.uint8)
        cv2.putText(dummy, "ArUco Detection Controls", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(dummy, "Adjust trackbars below", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(dummy, "Press 's' to save parameters", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(dummy, "Press 'q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow('Controls', dummy)
        
        # Create trackbars for all parameters
        cv2.createTrackbar('AdaptThresh WinMin', 'Controls', 
                          self.params['adaptiveThreshWinSizeMin'], 50, self.update_detection)
        cv2.createTrackbar('AdaptThresh WinMax', 'Controls', 
                          self.params['adaptiveThreshWinSizeMax'], 100, self.update_detection)
        cv2.createTrackbar('AdaptThresh Step', 'Controls', 
                          self.params['adaptiveThreshWinSizeStep'], 20, self.update_detection)
        cv2.createTrackbar('AdaptThresh Const', 'Controls', 
                          self.params['adaptiveThreshConstant'], 50, self.update_detection)
        
        cv2.createTrackbar('Min Perimeter (*0.01)', 'Controls', 
                          self.params['minMarkerPerimeterRate'], 20, self.update_detection)
        cv2.createTrackbar('Max Perimeter (*0.01)', 'Controls', 
                          self.params['maxMarkerPerimeterRate'], 1000, self.update_detection)
        cv2.createTrackbar('Polygon Accuracy (*0.01)', 'Controls', 
                          self.params['polygonalApproxAccuracyRate'], 20, self.update_detection)
        cv2.createTrackbar('Min Corner Dist (*0.01)', 'Controls', 
                          self.params['minCornerDistanceRate'], 20, self.update_detection)
        cv2.createTrackbar('Min Dist Border', 'Controls', 
                          self.params['minDistanceToBorder'], 20, self.update_detection)
        
        cv2.createTrackbar('Border Bits', 'Controls', 
                          self.params['markerBorderBits'], 5, self.update_detection)
        cv2.createTrackbar('Min Otsu StdDev (*0.1)', 'Controls', 
                          self.params['minOtsuStdDev'], 200, self.update_detection)
        cv2.createTrackbar('Perspective Pixels', 'Controls', 
                          self.params['perspectiveRemovePixelPerCell'], 20, self.update_detection)
        cv2.createTrackbar('Perspective Margin (*0.01)', 'Controls', 
                          self.params['perspectiveRemoveIgnoredMarginPerCell'], 50, self.update_detection)
        
        cv2.createTrackbar('Corner Refine Win', 'Controls', 
                          self.params['cornerRefinementWinSize'], 20, self.update_detection)
        cv2.createTrackbar('Corner Refine Iter', 'Controls', 
                          self.params['cornerRefinementMaxIterations'], 100, self.update_detection)
        cv2.createTrackbar('Corner Refine Acc (*0.01)', 'Controls', 
                          self.params['cornerRefinementMinAccuracy'], 100, self.update_detection)
        cv2.createTrackbar('Error Correction (*0.01)', 'Controls', 
                          self.params['errorCorrectionRate'], 100, self.update_detection)
        
        # Preprocessing parameters
        cv2.createTrackbar('Gaussian Blur', 'Controls', 
                          self.params['gaussianBlurKernel'], 21, self.update_detection)
        cv2.createTrackbar('CLAHE Clip (*0.1)', 'Controls', 
                          self.params['claheClipLimit'], 100, self.update_detection)
        cv2.createTrackbar('CLAHE Tile Size', 'Controls', 
                          self.params['claheTileSize'], 16, self.update_detection)
        cv2.createTrackbar('Bilateral D', 'Controls', 
                          self.params['bilateralD'], 20, self.update_detection)
        cv2.createTrackbar('Bilateral SigColor', 'Controls', 
                          self.params['bilateralSigmaColor'], 200, self.update_detection)
        cv2.createTrackbar('Bilateral SigSpace', 'Controls', 
                          self.params['bilateralSigmaSpace'], 200, self.update_detection)
        
        cv2.createTrackbar('Use Red Channel', 'Controls', 
                          self.params['useRedChannel'], 1, self.update_detection)
        cv2.createTrackbar('Contrast (*0.1)', 'Controls', 
                          self.params['contrastBrightness'], 30, self.update_detection)
        cv2.createTrackbar('Brightness', 'Controls', 
                          self.params['brightnessOffset'], 100, self.update_detection)
        
        # Add specialized controls for clay/silica challenges
        cv2.createTrackbar('White Saturation Clip', 'Controls', 
                          0, 100, self.update_detection)  # Clip bright white values
        cv2.createTrackbar('Shadow Boost', 'Controls', 
                          0, 100, self.update_detection)  # Boost dark areas
        cv2.createTrackbar('Red Tone Filter', 'Controls', 
                          0, 1, self.update_detection)  # Filter out red background
        cv2.createTrackbar('Histogram Equalize', 'Controls', 
                          0, 1, self.update_detection)  # Global histogram equalization
        cv2.createTrackbar('Morph Close Size', 'Controls', 
                          0, 10, self.update_detection)  # Fill gaps in markers
        cv2.createTrackbar('Edge Enhance', 'Controls', 
                          0, 100, self.update_detection)  # Enhance edges
        cv2.createTrackbar('Gamma Correction (*10)', 'Controls', 
                          10, 30, self.update_detection)  # Gamma correction (1.0 = 10)
    
    def load_image(self):
        """Load the image file"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"Could not load image: {self.image_path}")
            return False
        
        # Resize if too large
        height, width = self.original_image.shape[:2]
        if width > 1200 or height > 900:
            scale = min(1200/width, 900/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        return True
    
    def get_current_parameters(self):
        """Get current parameter values from trackbars"""
        parameters = cv2.aruco.DetectorParameters()
        
        # Get values from trackbars and convert to proper ranges
        parameters.adaptiveThreshWinSizeMin = cv2.getTrackbarPos('AdaptThresh WinMin', 'Controls')
        parameters.adaptiveThreshWinSizeMax = cv2.getTrackbarPos('AdaptThresh WinMax', 'Controls')
        parameters.adaptiveThreshWinSizeStep = cv2.getTrackbarPos('AdaptThresh Step', 'Controls')
        parameters.adaptiveThreshConstant = cv2.getTrackbarPos('AdaptThresh Const', 'Controls')
        
        parameters.minMarkerPerimeterRate = cv2.getTrackbarPos('Min Perimeter (*0.01)', 'Controls') * 0.01
        parameters.maxMarkerPerimeterRate = cv2.getTrackbarPos('Max Perimeter (*0.01)', 'Controls') * 0.01
        parameters.polygonalApproxAccuracyRate = cv2.getTrackbarPos('Polygon Accuracy (*0.01)', 'Controls') * 0.01
        parameters.minCornerDistanceRate = cv2.getTrackbarPos('Min Corner Dist (*0.01)', 'Controls') * 0.01
        parameters.minDistanceToBorder = cv2.getTrackbarPos('Min Dist Border', 'Controls')
        
        parameters.markerBorderBits = cv2.getTrackbarPos('Border Bits', 'Controls')
        parameters.minOtsuStdDev = cv2.getTrackbarPos('Min Otsu StdDev (*0.1)', 'Controls') * 0.1
        parameters.perspectiveRemovePixelPerCell = cv2.getTrackbarPos('Perspective Pixels', 'Controls')
        parameters.perspectiveRemoveIgnoredMarginPerCell = cv2.getTrackbarPos('Perspective Margin (*0.01)', 'Controls') * 0.01
        
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = cv2.getTrackbarPos('Corner Refine Win', 'Controls')
        parameters.cornerRefinementMaxIterations = cv2.getTrackbarPos('Corner Refine Iter', 'Controls')
        parameters.cornerRefinementMinAccuracy = cv2.getTrackbarPos('Corner Refine Acc (*0.01)', 'Controls') * 0.01
        parameters.errorCorrectionRate = cv2.getTrackbarPos('Error Correction (*0.01)', 'Controls') * 0.01
        
        # Ensure minimum values
        if parameters.adaptiveThreshWinSizeMin < 3:
            parameters.adaptiveThreshWinSizeMin = 3
        if parameters.adaptiveThreshWinSizeMax < parameters.adaptiveThreshWinSizeMin:
            parameters.adaptiveThreshWinSizeMax = parameters.adaptiveThreshWinSizeMin + 2
        if parameters.cornerRefinementWinSize < 1:
            parameters.cornerRefinementWinSize = 1
        if parameters.cornerRefinementMaxIterations < 1:
            parameters.cornerRefinementMaxIterations = 1
        
        return parameters
    
    def preprocess_image(self, img):
        """Preprocess image with current parameters - specialized for clay/silica challenges"""
        # Get preprocessing parameters
        use_red = cv2.getTrackbarPos('Use Red Channel', 'Controls')
        blur_kernel = cv2.getTrackbarPos('Gaussian Blur', 'Controls')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip (*0.1)', 'Controls') * 0.1
        clahe_tile = cv2.getTrackbarPos('CLAHE Tile Size', 'Controls')
        bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Controls')
        bilateral_color = cv2.getTrackbarPos('Bilateral SigColor', 'Controls')
        bilateral_space = cv2.getTrackbarPos('Bilateral SigSpace', 'Controls')
        contrast = cv2.getTrackbarPos('Contrast (*0.1)', 'Controls') * 0.1
        brightness = cv2.getTrackbarPos('Brightness', 'Controls') - 50
        
        # New specialized parameters
        white_clip = cv2.getTrackbarPos('White Saturation Clip', 'Controls')
        shadow_boost = cv2.getTrackbarPos('Shadow Boost', 'Controls')
        red_filter = cv2.getTrackbarPos('Red Tone Filter', 'Controls')
        hist_eq = cv2.getTrackbarPos('Histogram Equalize', 'Controls')
        morph_size = cv2.getTrackbarPos('Morph Close Size', 'Controls')
        edge_enhance = cv2.getTrackbarPos('Edge Enhance', 'Controls')
        gamma_val = cv2.getTrackbarPos('Gamma Correction (*10)', 'Controls') / 10.0
        
        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 1:
            blur_kernel = 1
        
        # Ensure tile size is positive
        if clahe_tile < 1:
            clahe_tile = 1
        
        # STEP 1: Channel selection and red tone filtering
        if red_filter:
            # Create a mask to filter out red-dominant areas (clay background)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Define red range in HSV (handles clay color around 180,80,53)
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Use green-blue channels more for red areas
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            green_blue = cv2.addWeighted(img[:,:,1], 0.6, img[:,:,0], 0.4, 0)
            
            # Blend based on red mask
            processed = np.where(red_mask[..., np.newaxis] > 0, 
                               green_blue[..., np.newaxis], 
                               gray[..., np.newaxis]).squeeze()
        elif use_red:
            processed = img[:, :, 2]  # Red channel (BGR format)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # STEP 2: Handle white saturation from silica
        if white_clip > 0:
            # Clip extremely bright values and redistribute
            clip_threshold = 255 - (white_clip * 2)  # 0-100 -> 255-55
            mask_bright = processed > clip_threshold
            processed[mask_bright] = clip_threshold
            
            # Optionally stretch the remaining range
            if np.max(processed) > 0:
                processed = (processed / np.max(processed) * 255).astype(np.uint8)
        
        # STEP 3: Gamma correction for non-linear brightness adjustment
        if gamma_val != 1.0:
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed = cv2.LUT(processed, table)
        
        # STEP 4: Shadow boosting for dark engraved areas
        if shadow_boost > 0:
            # Create mask for dark areas (likely engraved)
            dark_threshold = 80  # Adjust as needed
            dark_mask = processed < dark_threshold
            
            # Boost dark areas more aggressively
            boost_factor = 1.0 + (shadow_boost / 100.0)
            processed_boosted = processed.astype(np.float32) * boost_factor
            processed_boosted = np.clip(processed_boosted, 0, 255).astype(np.uint8)
            
            # Apply boost only to dark areas
            processed = np.where(dark_mask, processed_boosted, processed)
        
        # STEP 5: Apply contrast and brightness
        if contrast != 1.0 or brightness != 0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        
        # STEP 6: Histogram equalization for better contrast distribution
        if hist_eq:
            processed = cv2.equalizeHist(processed)
        
        # STEP 7: Apply Gaussian blur
        if blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # STEP 8: Apply CLAHE (local contrast enhancement)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
            processed = clahe.apply(processed)
        
        # STEP 9: Edge enhancement for better marker detection
        if edge_enhance > 0:
            # Create edge enhancement kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            edge_factor = edge_enhance / 100.0
            enhanced = cv2.filter2D(processed, -1, kernel * edge_factor)
            processed = cv2.addWeighted(processed, 1.0, enhanced, edge_factor, 0)
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # STEP 10: Apply bilateral filter
        if bilateral_d > 0:
            processed = cv2.bilateralFilter(processed, bilateral_d, bilateral_color, bilateral_space)
        
        # STEP 11: Morphological closing to fill gaps in markers
        if morph_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size*2+1, morph_size*2+1))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def update_detection(self, val):
        """Update detection with current parameters"""
        if self.original_image is None:
            return
        
        # Get current parameters
        parameters = self.get_current_parameters()
        
        # Preprocess image
        processed = self.preprocess_image(self.original_image)
        
        # Convert to BGR for ArUco detection
        if len(processed.shape) == 2:
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_bgr = processed
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed_bgr, self.aruco_dict, parameters=parameters
        )
        
        # Create result image
        result_img = self.original_image.copy()
        info_text = []
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
            info_text.append(f"Detected: {len(ids)} markers")
            info_text.append(f"IDs: {ids.flatten()}")
        else:
            info_text.append("No markers detected")
        
        info_text.append(f"Rejected: {len(rejected)} candidates")
        
        # Add text overlay
        for i, text in enumerate(info_text):
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display images
        cv2.imshow('Original', self.original_image)
        cv2.imshow('Detection Result', result_img)
        cv2.imshow('Processed', processed)
        
        # Print detection info
        if ids is not None:
            print(f"\nDetected {len(ids)} markers: {ids.flatten()}")
            for i, corner_set in enumerate(corners):
                marker_id = ids[i][0]
                corners_text = ", ".join([f"({c[0]:.1f},{c[1]:.1f})" for c in corner_set[0]])
                print(f"Marker {marker_id}: {corners_text}")
    
    def save_current_parameters(self):
        """Save current parameters to file"""
        params_dict = {}
        trackbar_names = [
            'AdaptThresh WinMin', 'AdaptThresh WinMax', 'AdaptThresh Step', 'AdaptThresh Const',
            'Min Perimeter (*0.01)', 'Max Perimeter (*0.01)', 'Polygon Accuracy (*0.01)', 
            'Min Corner Dist (*0.01)', 'Min Dist Border', 'Border Bits', 'Min Otsu StdDev (*0.1)',
            'Perspective Pixels', 'Perspective Margin (*0.01)', 'Corner Refine Win', 
            'Corner Refine Iter', 'Corner Refine Acc (*0.01)', 'Error Correction (*0.01)',
            'Gaussian Blur', 'CLAHE Clip (*0.1)', 'CLAHE Tile Size', 'Bilateral D',
            'Bilateral SigColor', 'Bilateral SigSpace', 'Use Red Channel', 
            'Contrast (*0.1)', 'Brightness'
        ]
        
        for name in trackbar_names:
            params_dict[name] = cv2.getTrackbarPos(name, 'Controls')
        
        with open('aruco_parameters.txt', 'w') as f:
            for name, value in params_dict.items():
                f.write(f"{name}: {value}\n")
        
        print("Parameters saved to 'aruco_parameters.txt'")
    
    def run(self):
        """Run the interactive detector"""
        if not self.load_image():
            return
        
        print("Interactive ArUco Detector")
        print("Controls:")
        print("- Look for the 'Controls' window with trackbars")
        print("- Adjust trackbars to change detection parameters")
        print("- Press 's' to save current parameters")
        print("- Press 'r' to reset to defaults")
        print("- Press 'q' to quit")
        print("\nIf you don't see the Controls window, try:")
        print("- Check your taskbar for additional OpenCV windows")
        print("- Try Alt+Tab to cycle through windows")
        print("- The window might be behind other windows")
        
        # Make sure all windows are visible
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
        
        # Position windows so they don't overlap
        cv2.moveWindow('Original', 600, 50)
        cv2.moveWindow('Detection Result', 600, 400)
        cv2.moveWindow('Processed', 1100, 50)
        
        # Initial detection
        self.update_detection(0)
        
        # Make sure Controls window stays on top initially
        cv2.setWindowProperty('Controls', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(100)  # Brief pause
        cv2.setWindowProperty('Controls', cv2.WND_PROP_TOPMOST, 0)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_current_parameters()
            elif key == ord('r'):
                # Reset to default values
                for param_name, default_value in self.params.items():
                    trackbar_name = self.get_trackbar_name(param_name)
                    if trackbar_name:
                        cv2.setTrackbarPos(trackbar_name, 'Controls', default_value)
                self.update_detection(0)
            elif key == ord('h'):
                # Help - bring Controls window to front
                cv2.setWindowProperty('Controls', cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(100)
                cv2.setWindowProperty('Controls', cv2.WND_PROP_TOPMOST, 0)
                print("Brought Controls window to front")
        
        cv2.destroyAllWindows()
    
    def get_trackbar_name(self, param_name):
        """Map parameter names to trackbar names"""
        mapping = {
            'adaptiveThreshWinSizeMin': 'AdaptThresh WinMin',
            'adaptiveThreshWinSizeMax': 'AdaptThresh WinMax',
            'adaptiveThreshWinSizeStep': 'AdaptThresh Step',
            'adaptiveThreshConstant': 'AdaptThresh Const',
            'minMarkerPerimeterRate': 'Min Perimeter (*0.01)',
            'maxMarkerPerimeterRate': 'Max Perimeter (*0.01)',
            'polygonalApproxAccuracyRate': 'Polygon Accuracy (*0.01)',
            'minCornerDistanceRate': 'Min Corner Dist (*0.01)',
            'minDistanceToBorder': 'Min Dist Border',
            'markerBorderBits': 'Border Bits',
            'minOtsuStdDev': 'Min Otsu StdDev (*0.1)',
            'perspectiveRemovePixelPerCell': 'Perspective Pixels',
            'perspectiveRemoveIgnoredMarginPerCell': 'Perspective Margin (*0.01)',
            'cornerRefinementWinSize': 'Corner Refine Win',
            'cornerRefinementMaxIterations': 'Corner Refine Iter',
            'cornerRefinementMinAccuracy': 'Corner Refine Acc (*0.01)',
            'errorCorrectionRate': 'Error Correction (*0.01)',
            'gaussianBlurKernel': 'Gaussian Blur',
            'claheClipLimit': 'CLAHE Clip (*0.1)',
            'claheTileSize': 'CLAHE Tile Size',
            'bilateralD': 'Bilateral D',
            'bilateralSigmaColor': 'Bilateral SigColor',
            'bilateralSigmaSpace': 'Bilateral SigSpace',
            'useRedChannel': 'Use Red Channel',
            'contrastBrightness': 'Contrast (*0.1)',
            'brightnessOffset': 'Brightness'
        }
        return mapping.get(param_name)

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
    # Create test markers
    create_test_marker(0)
    create_test_marker(1)
    
    # Run interactive detector
    detector = InteractiveArucoDetector('azulejo3.jpg')
    detector.run()