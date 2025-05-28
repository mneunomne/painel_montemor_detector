"""
Refactored ArUco detector for character recognition on clay surfaces
"""

import cv2
import numpy as np
import os
import glob
# from character_recognition import recognize_character_from_image, join_characters_to_message


class ArucoDetector:
    def __init__(self, image_path='image.jpg'):
        """ArUco detector with optimized parameters for red clay surfaces"""
        self.image_path = image_path
        self.original_image = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detection_params = self._get_detection_parameters()
        self.load_image()
    
    def _get_detection_parameters(self):
        """Get optimized ArUco detection parameters"""
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 40
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.minCornerDistanceRate = 0.05
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.errorCorrectionRate = 0.6
        return params
    
    def load_image(self):
        """Load and resize image if needed"""
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
    
    def preprocess_image(self, img):
        """Preprocess image for better detection"""
        # Use red channel for clay surfaces
        processed = img[:, :, 2]  # Red channel (BGR format)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        
        # Apply bilateral filter to reduce noise while preserving edges
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        return processed
    
    def detect_markers(self):
        """Detect ArUco markers in the image"""
        if self.original_image is None:
            print("No image loaded!")
            return None, None, None
        
        # Preprocess and detect
        processed = self.preprocess_image(self.original_image)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed_bgr, self.aruco_dict, parameters=self.detection_params
        )
        
        # Create result visualization
        result_img = self.original_image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
            print(f"Detected {len(ids)} markers: {ids.flatten()}")
        else:
            print("No markers detected")
        
        return corners, ids, result_img
    
    def get_perspective_transform(self, corners, ids):
        """Get perspective transformation matrix from detected markers"""
        if ids is None or len(ids) < 4:
            print("Not enough markers for perspective transform")
            return None, None
        
        # Map marker IDs to positions
        id_to_position = {24: 'top-left', 25: 'top-right', 26: 'bottom-left', 27: 'bottom-right'}
        corner_mapping = {'top-right': 1, 'top-left': 0, 'bottom-right': 2, 'bottom-left': 3}
        
        positions = {}
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]
            marker_id_val = int(marker_id[0])
            
            if marker_id_val in id_to_position:
                position_name = id_to_position[marker_id_val]
                corner_index = corner_mapping[position_name]
                corner_point = marker_corners[corner_index]
                positions[position_name] = {'x': int(corner_point[0]), 'y': int(corner_point[1])}
        
        if len(positions) != 4:
            print(f"Found {len(positions)} out of 4 required corner markers")
            return None, None
        
        # Apply padding and create transformation
        padding = {'top-left': (-8, -6), 'top-right': (8, -6), 
                  'bottom-right': (8, 8), 'bottom-left': (-8, 8)}
        
        src_pts = []
        for corner in ['top-left', 'top-right', 'bottom-right', 'bottom-left']:
            x = positions[corner]['x'] + padding[corner][0]
            y = positions[corner]['y'] + padding[corner][1]
            src_pts.append([x, y])
        
        src_pts = np.array(src_pts, dtype=np.float32)
        
        width = height = 600
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix, (width, height)
    
    def load_templates(self, template_folder='character_templates'):
        """Load character templates from folder"""
        templates = {}
        
        if not os.path.exists(template_folder):
            print(f"Template folder '{template_folder}' not found!")
            return templates
        
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for template_path in glob.glob(os.path.join(template_folder, ext)):
                template_name = os.path.splitext(os.path.basename(template_path))[0]
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                
                if template_img is not None:
                    templates[template_name] = template_img
        
        print(f"Loaded {len(templates)} templates")
        return templates
    
    def process_grid(self, warped_img, templates=None, data_length=None, export_cells=False):
        """Process the warped image grid and recognize characters"""
        height, width = warped_img.shape[:2]
        
        # Grid parameters
        grid_rows = grid_cols = 7
        cell_width = cell_height = 62
        start_x, start_y = 4, 3
        gap = int(((width - start_x) - (grid_rows * cell_width)) / (grid_rows - 1))
        
        # Setup export folder if needed
        if export_cells:
            export_folder = 'exported_cells'
            os.makedirs(export_folder, exist_ok=True)
            # Clear existing files
            for f in glob.glob(os.path.join(export_folder, '*.png')):
                os.remove(f)
        
        # Process each cell
        grid_results = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        result_img = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR) if len(warped_img.shape) == 2 else warped_img.copy()
        
        cell_count = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Skip corner cells
                if ((row == 0 and col == 0) or (row == 0 and col == grid_cols - 1) or 
                    (row == grid_rows - 1 and col == 0) or (row == grid_rows - 1 and col == grid_cols - 1)):
                    continue
                
                if data_length and cell_count >= data_length:
                    break
                
                # Extract cell
                cell_x = start_x + col * (cell_width + gap)
                cell_y = start_y + row * (cell_height + gap)
                cell_img = warped_img[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width]
                
                # Export cell if requested
                if export_cells:
                    cell_filename = os.path.join(export_folder, f"cell_{row}_{col}.png")
                    cv2.imwrite(cell_filename, cell_img)
                
                # Recognize character
                character, confidence = recognize_character_from_image(cell_img, templates)
                
                if character and character != '?':
                    grid_results[row][col] = {
                        'character': character,
                        'confidence': confidence,
                        'position': (cell_x, cell_y)
                    }
                    
                    # Draw result
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                    cv2.rectangle(result_img, (cell_x, cell_y), 
                                (cell_x + cell_width, cell_y + cell_height), color, 2)
                    cv2.putText(result_img, character, (cell_x + 2, cell_y + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(result_img, f"{confidence:.2f}", (cell_x + 2, cell_y + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
                    print(f"Cell ({row},{col}): '{character}' (confidence: {confidence:.3f})")
                else:
                    cv2.rectangle(result_img, (cell_x, cell_y), 
                                (cell_x + cell_width, cell_y + cell_height), (0, 0, 255), 1)
                
                cell_count += 1
        
        # Generate complete message
        message = join_characters_to_message(grid_results, grid_rows, grid_cols)
        print(f"\nDecoded message: '{message}'")
        
        return grid_results, result_img, message