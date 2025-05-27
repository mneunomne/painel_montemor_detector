import cv2
import numpy as np

class ArucoDetector:
    def __init__(self, image_path='image.jpg'):
        """
        ArUco detector with optimized parameters for red clay surfaces
        """
        self.image_path = image_path
        self.original_image = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Optimized parameters for red clay detection
        self.params = {
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 23,
            'adaptiveThreshWinSizeStep': 10,
            'adaptiveThreshConstant': 0,
            'minMarkerPerimeterRate': 0.03,
            'maxMarkerPerimeterRate': 4.0,
            'polygonalApproxAccuracyRate': 0.03,
            'minCornerDistanceRate': 0.05,
            'minDistanceToBorder': 3,
            'markerBorderBits': 1,
            'minOtsuStdDev': 5.0,
            'perspectiveRemovePixelPerCell': 8,
            'perspectiveRemoveIgnoredMarginPerCell': 0.13,
            'cornerRefinementWinSize': 5,
            'cornerRefinementMaxIterations': 30,
            'cornerRefinementMinAccuracy': 0.1,
            'errorCorrectionRate': 0.6,
            # Preprocessing parameters
            'gaussianBlurKernel': 2,
            'claheClipLimit': 2.0,
            'claheTileSize': 8,
            'bilateralD': 9,
            'bilateralSigmaColor': 75,
            'bilateralSigmaSpace': 75,
            'useRedChannel': True,
            'contrastBrightness': 1.0,
            'brightnessOffset': 0,
            # Specialized parameters for clay/silica challenges
            'whiteSaturationClip': 0,
            'shadowBoost': 0,
            'redToneFilter': False,
            'histogramEqualize': False,
            'morphCloseSize': 0,
            'edgeEnhance': 0,
            'gammaCorrection': 1.0
        }
        
        self.load_image()
    
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
    
    def get_detection_parameters(self):
        """Get ArUco detection parameters"""
        parameters = cv2.aruco.DetectorParameters()
        
        parameters.adaptiveThreshWinSizeMin = self.params['adaptiveThreshWinSizeMin']
        parameters.adaptiveThreshWinSizeMax = self.params['adaptiveThreshWinSizeMax']
        parameters.adaptiveThreshWinSizeStep = self.params['adaptiveThreshWinSizeStep']
        parameters.adaptiveThreshConstant = self.params['adaptiveThreshConstant']
        
        parameters.minMarkerPerimeterRate = self.params['minMarkerPerimeterRate']
        parameters.maxMarkerPerimeterRate = self.params['maxMarkerPerimeterRate']
        parameters.polygonalApproxAccuracyRate = self.params['polygonalApproxAccuracyRate']
        parameters.minCornerDistanceRate = self.params['minCornerDistanceRate']
        parameters.minDistanceToBorder = self.params['minDistanceToBorder']
        
        parameters.markerBorderBits = self.params['markerBorderBits']
        parameters.minOtsuStdDev = self.params['minOtsuStdDev']
        parameters.perspectiveRemovePixelPerCell = self.params['perspectiveRemovePixelPerCell']
        parameters.perspectiveRemoveIgnoredMarginPerCell = self.params['perspectiveRemoveIgnoredMarginPerCell']
        
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = self.params['cornerRefinementWinSize']
        parameters.cornerRefinementMaxIterations = self.params['cornerRefinementMaxIterations']
        parameters.cornerRefinementMinAccuracy = self.params['cornerRefinementMinAccuracy']
        parameters.errorCorrectionRate = self.params['errorCorrectionRate']
        
        return parameters
    
    def preprocess_image(self, img):
        """Preprocess image - specialized for clay/silica challenges"""
        # Get preprocessing parameters
        use_red = self.params['useRedChannel']
        blur_kernel = self.params['gaussianBlurKernel']
        clahe_clip = self.params['claheClipLimit']
        clahe_tile = self.params['claheTileSize']
        bilateral_d = self.params['bilateralD']
        bilateral_color = self.params['bilateralSigmaColor']
        bilateral_space = self.params['bilateralSigmaSpace']
        contrast = self.params['contrastBrightness']
        brightness = self.params['brightnessOffset']
        
        # Specialized parameters
        white_clip = self.params['whiteSaturationClip']
        shadow_boost = self.params['shadowBoost']
        red_filter = self.params['redToneFilter']
        hist_eq = self.params['histogramEqualize']
        morph_size = self.params['morphCloseSize']
        edge_enhance = self.params['edgeEnhance']
        gamma_val = self.params['gammaCorrection']
        
        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 1:
            blur_kernel = 1
        
        # Ensure tile size is positive
        if clahe_tile < 1:
            clahe_tile = 1
        
        elif use_red:
            processed = img[:, :, 2]  # Red channel (BGR format)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # STEP 5: Apply contrast and brightness
        if contrast != 1.0 or brightness != 0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        
        # STEP 7: Apply Gaussian blur
        if blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # STEP 8: Apply CLAHE (local contrast enhancement)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
            processed = clahe.apply(processed)
        
        # STEP 10: Apply bilateral filter
        if bilateral_d > 0:
            processed = cv2.bilateralFilter(processed, bilateral_d, bilateral_color, bilateral_space)
        
        return processed
    
    def detect_markers(self, save_result=False, show_result=False):
        """
        Detect ArUco markers in the loaded image
        
        Args:
            save_result (bool): Whether to save the result image
            show_result (bool): Whether to display the result
            
        Returns:
            tuple: (corners, ids, rejected, result_image)
        """
        if self.original_image is None:
            print("No image loaded!")
            return None, None, None, None
        
        # Get detection parameters
        parameters = self.get_detection_parameters()
        
        # Preprocess image
        processed = self.preprocess_image(self.original_image)
        
        # Convert to BGR for ArUco detection if needed
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
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
            print(f"Detected {len(ids)} markers: {ids.flatten()}")
            
            # Print detailed corner information
            for i, corner_set in enumerate(corners):
                marker_id = ids[i][0]
                corners_text = ", ".join([f"({c[0]:.1f},{c[1]:.1f})" for c in corner_set[0]])
                print(f"Marker {marker_id}: {corners_text}")
        else:
            print("No markers detected")
        
        print(f"Rejected candidates: {len(rejected)}")
        
        # Add text overlay to result image
        info_text = []
        if ids is not None:
            info_text.append(f"Detected: {len(ids)} markers")
            info_text.append(f"IDs: {ids.flatten()}")
        else:
            info_text.append("No markers detected")
        
        info_text.append(f"Rejected: {len(rejected)} candidates")
        
        for i, text in enumerate(info_text):
            cv2.putText(result_img, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show result if requested
        if show_result:
            cv2.imshow('Detection Result', result_img)
            cv2.imshow('Processed Image', processed)
            print("Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return corners, ids, rejected, result_img
    
    def update_parameters(self, **kwargs):
        """
        Update detection parameters
        
        Args:
            **kwargs: Parameter name-value pairs to update
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"Updated {key}: {value}")
            else:
                print(f"Unknown parameter: {key}")
    
    def get_parameters(self):
        """Get current parameters"""
        return self.params.copy()
    
    def save_parameters(self, filename='aruco_parameters.txt'):
        """Save current parameters to file"""
        with open(filename, 'w') as f:
            f.write("ArUco Detection Parameters\n")
            f.write("=" * 30 + "\n\n")
            
            # Detection parameters
            f.write("Detection Parameters:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.params.items():
                if key not in ['gaussianBlurKernel', 'claheClipLimit', 'claheTileSize', 
                              'bilateralD', 'bilateralSigmaColor', 'bilateralSigmaSpace',
                              'useRedChannel', 'contrastBrightness', 'brightnessOffset',
                              'whiteSaturationClip', 'shadowBoost', 'redToneFilter',
                              'histogramEqualize', 'morphCloseSize', 'edgeEnhance', 'gammaCorrection']:
                    f.write(f"{key}: {value}\n")
            
            f.write("\nPreprocessing Parameters:\n")
            f.write("-" * 25 + "\n")
            preprocessing_params = ['gaussianBlurKernel', 'claheClipLimit', 'claheTileSize', 
                                  'bilateralD', 'bilateralSigmaColor', 'bilateralSigmaSpace',
                                  'useRedChannel', 'contrastBrightness', 'brightnessOffset',
                                  'whiteSaturationClip', 'shadowBoost', 'redToneFilter',
                                  'histogramEqualize', 'morphCloseSize', 'edgeEnhance', 'gammaCorrection']
            
            for key in preprocessing_params:
                if key in self.params:
                    f.write(f"{key}: {self.params[key]}\n")
        
        print(f"Parameters saved to '{filename}'")

def create_test_marker(marker_id=0, size=200):
    """Create a test ArUco marker for testing"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add white border for better detection
    bordered = cv2.copyMakeBorder(marker_img, 50, 50, 50, 50, 
                                  cv2.BORDER_CONSTANT, value=255)
    
    filename = f'test_marker_{marker_id}.png'
    cv2.imwrite(filename, bordered)
    print(f"Test marker {marker_id} saved as '{filename}'")
    
    return bordered

if __name__ == "__main__":
    # Example usage
    
    # Initialize detector
    detector = ArucoDetector('azulejo5.jpg')
    
    # Detect markers with default parameters
    print("=== Detection with default parameters ===")
    corners, ids, rejected, result = detector.detect_markers(save_result=True, show_result=True)

    # Map marker IDs to positions and their corresponding corner indices
    positions = {}
    id_to_position = {10: 'top-right', 12: 'top-left', 11: 'bottom-right', 13: 'bottom-left'}

    # ArUco markers have 4 corners in clockwise order starting from top-left:
    # Corner 0: top-left of marker
    # Corner 1: top-right of marker  
    # Corner 2: bottom-right of marker
    # Corner 3: bottom-left of marker

    # Map each ROI corner to the corresponding marker corner
    corner_mapping = {
        'top-right': 0,     # Use top-left corner of top-left marker
        'top-left': 3,    # Use top-right corner of top-right marker
        'bottom-right': 1, # Use bottom-right corner of bottom-right marker
        'bottom-left': 2   # Use bottom-left corner of bottom-left marker
    }

    avarage_width = 0

    if ids is not None:
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]  # Get the 4 corners of this marker
            
            marker_id_val = int(marker_id[0])
            if marker_id_val in id_to_position:
                position_name = id_to_position[marker_id_val]
                corner_index = corner_mapping[position_name]
                
                # Get the specific corner point for this ROI corner
                corner_point = marker_corners[corner_index]
                
                positions[position_name] = {
                    'x': int(corner_point[0]), 
                    'y': int(corner_point[1]),
                    'corners': marker_corners,
                    'width': int(np.linalg.norm(marker_corners[0] - marker_corners[1])),
                }

        print(f"Detected positions: {positions}")

        if len(positions) == 4:
            # Calculate the center of all corner points (not marker centers)
            corner_points = [(pos['x'], pos['y']) for pos in positions.values()]
            avg_center = (int(np.mean([c[0] for c in corner_points])), int(np.mean([c[1] for c in corner_points])))
            avarage_width = int(np.mean([pos['width'] for pos in positions.values()]))
            print(f"Average Width: {avarage_width}")
            
            padding_config = {
            'top-left': {'x': -12, 'y': -12},        # Move left and up
            'top-right': {'x': 12, 'y': -12},        # Move right and up
            'bottom-right': {'x': 12, 'y': 12},      # Move right and down
            'bottom-left': {'x': -12, 'y': 12}       # Move left and down
        }
        
        # Alternative: uniform padding for all corners
        # uniform_padding = 10
        # padding_config = {
        #     'top-left': {'x': -uniform_padding, 'y': -uniform_padding},
        #     'top-right': {'x': uniform_padding, 'y': -uniform_padding},
        #     'bottom-right': {'x': uniform_padding, 'y': uniform_padding},
        #     'bottom-left': {'x': -uniform_padding, 'y': uniform_padding}
        # }
        
        # Calculate padded corners
        padded_corners = {}
        for corner_name in ['top-left', 'top-right', 'bottom-right', 'bottom-left']:
            if corner_name in positions:
                original_x = positions[corner_name]['x']
                original_y = positions[corner_name]['y']
                pad_x = padding_config[corner_name]['x']
                pad_y = padding_config[corner_name]['y']
                
                padded_corners[corner_name] = {
                    'x': original_x + pad_x,
                    'y': original_y + pad_y
                }
        
        # Create corner points for drawing and transformation
        tl = (padded_corners['top-left']['x'], padded_corners['top-left']['y'])
        tr = (padded_corners['top-right']['x'], padded_corners['top-right']['y'])
        br = (padded_corners['bottom-right']['x'], padded_corners['bottom-right']['y'])
        bl = (padded_corners['bottom-left']['x'], padded_corners['bottom-left']['y'])
        
        # Draw polygon connecting the padded ROI corners
        roi_corners = np.array([tl, tr, br, bl], dtype=np.int32)
        #cv2.polylines(result, [roi_corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw the original fiducial corners for reference
        original_corners = np.array([
            (positions['top-left']['x'], positions['top-left']['y']),
            (positions['top-right']['x'], positions['top-right']['y']),
            (positions['bottom-right']['x'], positions['bottom-right']['y']),
            (positions['bottom-left']['x'], positions['bottom-left']['y'])
        ], dtype=np.int32)
        #cv2.polylines(result, [original_corners], isClosed=True, color=(255, 0, 0), thickness=1)
        
        # Mark each original corner point
        for pos_name, pos_data in positions.items():
            cv2.circle(result, (pos_data['x'], pos_data['y']), 4, (0, 0, 255), -1)
            cv2.putText(result, f"{pos_name} (orig)", (pos_data['x'] + 10, pos_data['y'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Mark each padded corner point
        for corner_name, corner_data in padded_corners.items():
            cv2.circle(result, (corner_data['x'], corner_data['y']), 6, (0, 255, 0), -1)
            cv2.putText(result, f"{corner_name} (pad)", (corner_data['x'] + 10, corner_data['y'] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Calculate the center of the padded ROI
        padded_points = [(corner['x'], corner['y']) for corner in padded_corners.values()]
        roi_center = (int(np.mean([p[0] for p in padded_points])), int(np.mean([p[1] for p in padded_points])))
        cv2.circle(result, roi_center, 8, (255, 255, 0), -1)
        cv2.putText(result, "ROI Center", (roi_center[0] - 30, roi_center[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Extract ROI using perspective warp with the padded corner points
        src_pts = np.array([
            [tl[0], tl[1]],     # top-left
            [tr[0], tr[1]],     # top-right
            [br[0], br[1]],     # bottom-right
            [bl[0], bl[1]]      # bottom-left
        ], dtype=np.float32)
        
        # Define destination rectangle (square output)
        width = height = 600
        
        # Map to a square with proper orientation
        dst_pts = np.array([
            [0, 0],           # top-left
            [width, 0],       # top-right
            [width, height],  # bottom-right
            [0, height]       # bottom-left
        ], dtype=np.float32)
        
        # Get perspective transform matrix and apply warp
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(result, matrix, (width, height))

        # Grid parameters
        grid_rows = 9
        grid_cols = 9
        cell_width = 39    # W: width of each grid cell
        cell_height = 39   # H: height of each grid cell
        gap = 30           # g: gap between grid cells
        
        # Calculate total grid dimensions
        total_grid_width = (grid_cols * cell_width) + ((grid_cols - 1) * gap)
        total_grid_height = (grid_rows * cell_height) + ((grid_rows - 1) * gap)
        
        # Calculate starting position to center the grid
        start_x = (width - total_grid_width) // 2
        start_y = (height - total_grid_height) // 2
        
        # Create a copy of warped image for grid overlay
        warped_with_grid = warped.copy()
        
        # Draw grid cells
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell position
                cell_x = start_x + col * (cell_width + gap)
                cell_y = start_y + row * (cell_height + gap)
                
                # Draw cell rectangle
                cv2.rectangle(warped_with_grid, 
                            (cell_x, cell_y), 
                            (cell_x + cell_width, cell_y + cell_height), 
                            (0, 255, 255), 2)  # Yellow grid lines
                
                # Optional: Add cell coordinates text (comment out if not needed)
                # cv2.putText(warped_with_grid, f"{row},{col}", 
                #            (cell_x + 2, cell_y + 15), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw grid outline
        cv2.rectangle(warped_with_grid, 
                    (start_x, start_y), 
                    (start_x + total_grid_width, start_y + total_grid_height), 
                    (255, 0, 255), 3)  # Magenta outline
        
        # Display grid information
        print(f"Applied padding:")
        for corner_name, pad_config in padding_config.items():
            if corner_name in positions:
                print(f"  {corner_name}: x={pad_config['x']:+d}, y={pad_config['y']:+d}")
        
        print(f"\nGrid configuration:")
        print(f"  Grid size: {grid_rows}x{grid_cols}")
        print(f"  Cell dimensions: {cell_width}x{cell_height}")
        print(f"  Gap size: {gap}")
        print(f"  Total grid size: {total_grid_width}x{total_grid_height}")
        print(f"  Grid position: ({start_x}, {start_y})")
        
        # Display the warped images
        cv2.imshow("Warped Perspective View", warped)
        cv2.imshow("Warped with Grid", warped_with_grid)
        
        # Display padding information
        print(f"Applied padding:")
        for corner_name, pad_config in padding_config.items():
            if corner_name in positions:
                print(f"  {corner_name}: x={pad_config['x']:+d}, y={pad_config['y']:+d}")
        
        # Display the warped image
        cv2.imshow("Warped Perspective View", warped)
                
    else:
        print(f"Not all markers detected. Found {len(positions)} markers:")
        for pos, data in positions.items():
            print(f"  {pos}: ({data['x']}, {data['y']})")

    # Display the original image with annotations
    cv2.imshow("Detected Markers", result)

    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()