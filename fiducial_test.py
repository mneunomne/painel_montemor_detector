import cv2
import numpy as np
import os
import glob


def get_patterns():
    """
    Return the pattern dictionary used for character encoding.
    """
    return {
        # Basic Latin letters - kept distinctive patterns
        'A': [[1,0,1], [1,1,1], [1,0,1]],      # Changed from original to be more A-like
        'Ã': [[0,1,0], [1,0,1], [1,1,1]],      # More distinctive from A
        'B': [[1,1,0], [1,1,1], [1,1,0]],      # More B-like with two bumps
        'C': [[1,1,1], [1,0,0], [1,1,1]],      # C-shape opening to right
        'Ç': [[1,1,1], [1,0,0], [1,1,0]],      # C with cedilla difference
        'D': [[1,1,0], [1,0,1], [1,1,0]],      # D-shape
        'E': [[1,1,1], [1,1,0], [1,1,1]],      # E with middle bar
        'É': [[0,1,0], [1,1,1], [1,0,0]],      # Distinctive from E
        'È': [[1,0,0], [1,1,1], [1,0,0]],      # Different accent pattern
        'Ê': [[0,1,0], [1,1,1], [1,1,1]],      # Hat-like accent
        'F': [[1,1,1], [1,1,0], [1,0,0]],      # F without bottom bar
        'G': [[1,1,1], [1,0,0], [1,0,1]],      # G with inner bar
        'H': [[1,0,1], [1,1,1], [1,0,1]],      # H-shape (kept as good)
        'I': [[1,1,1], [0,1,0], [1,1,1]],      # I with top/bottom bars
        'Í': [[0,0,1], [0,1,0], [1,1,1]],      # I with accent
        'J': [[0,1,1], [0,0,1], [1,0,1]],      # J hook shape
        'K': [[1,0,1], [1,1,0], [1,0,1]],      # K shape (kept)
        'L': [[1,0,0], [1,0,0], [1,1,1]],      # L shape (kept)
        'M': [[1,1,1], [1,0,1], [1,0,1]],      # M with peaks
        'N': [[1,1,0], [1,0,1], [0,1,1]],      # N diagonal (kept)
        'O': [[1,1,1], [1,0,1], [1,1,1]],      # O square (kept)
        'Õ': [[0,1,0], [1,0,1], [0,1,0]],      # O with tilde pattern
        'Ó': [[0,0,1], [1,0,1], [1,1,1]],      # O with accent
        'Ô': [[0,1,0], [1,0,1], [1,1,1]],      # O with hat
        'P': [[1,1,1], [1,1,0], [1,0,0]],      # P shape (kept)
        'Q': [[1,1,1], [1,0,1], [0,1,1]],      # Q with tail
        'R': [[1,1,0], [1,1,1], [1,0,1]],      # R shape (kept)
        'S': [[0,1,1], [0,1,0], [1,1,0]],      # S curve (kept)
        'T': [[1,1,1], [0,1,0], [0,1,0]],      # T shape (kept)
        'U': [[1,0,1], [1,0,1], [1,1,1]],      # U shape (kept)
        'Ú': [[0,0,1], [1,0,1], [0,1,0]],      # U with accent
        'V': [[1,0,1], [1,0,1], [0,1,0]],      # V shape (kept)
        'W': [[1,0,1], [1,0,1], [1,1,1]],      # W wide bottom
        'X': [[1,0,1], [0,1,0], [1,0,1]],      # X cross (kept)
        'Y': [[1,0,1], [0,1,0], [0,1,0]],      # Y shape (kept)
        'Z': [[1,1,1], [0,1,0], [1,1,1]],      # Z diagonal (kept)
        
        # Numbers - made more distinctive
        '0': [[1,1,1], [1,0,1], [1,1,1]],      # Square O (kept)
        '1': [[0,1,0], [1,1,0], [0,1,0]],      # Vertical line (kept)
        '2': [[1,1,1], [0,1,1], [1,1,1]],      # 2 shape (kept but was duplicate)
        '3': [[1,1,1], [0,1,1], [0,1,1]],      # Different from 2
        '4': [[1,0,1], [1,1,1], [0,0,1]],      # 4 shape (kept)
        '5': [[1,1,1], [1,1,0], [0,1,1]],      # Different from 6
        '6': [[1,1,1], [1,1,0], [1,0,1]],      # 6 with bottom gap
        '7': [[1,1,1], [0,0,1], [0,0,1]],      # 7 shape (kept)
        '8': [[1,1,1], [1,1,1], [1,1,1]],      # Full block (kept)
        '9': [[1,1,1], [1,1,1], [0,0,1]],      # 9 shape (kept)
        
        # Punctuation
        ' ': [[0,0,0], [0,0,0], [0,0,0]],      # Empty space
        '.': [[0,0,0], [0,0,0], [0,1,0]],      # Period (kept)
        ',': [[0,0,0], [0,0,0], [1,0,0]],      # Comma - different from period
        '-': [[0,0,0], [1,1,1], [0,0,0]],      # Hyphen (kept)
        '!': [[0,1,0], [0,1,0], [0,1,0]],      # Exclamation
        '?': [[1,1,1], [0,1,0], [0,1,0]],      # Question mark
        ':': [[0,1,0], [0,0,0], [0,1,0]],      # Colon
        ';': [[0,1,0], [0,0,0], [1,0,0]],      # Semicolon
        '|': [[0,0,0], [0,1,0], [0,0,0]],      # Separator
    }

class ArucoDetector:
    def __init__(self, image_path='image.jpg'):
        """ArUco detector with optimized parameters for red clay surfaces"""
        self.image_path = image_path
        self.original_image = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Optimized parameters for red clay detection
        self.params = {
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 40,
            'adaptiveThreshWinSizeStep': 10,
            'adaptiveThreshConstant': 0,
            'minMarkerPerimeterRate': 0.03,
            'maxMarkerPerimeterRate': 4.0,
            'polygonalApproxAccuracyRate': 0.03,
            'minCornerDistanceRate': 0.05,
            'minDistanceToBorder': 1,
            'markerBorderBits': 1,
            'minOtsuStdDev': 5.0,
            'perspectiveRemovePixelPerCell': 8,
            'perspectiveRemoveIgnoredMarginPerCell': 0.13,
            'cornerRefinementWinSize': 5,
            'cornerRefinementMaxIterations': 30,
            'cornerRefinementMinAccuracy': 0.1,
            'errorCorrectionRate': 0.6,
            'gaussianBlurKernel': 1,
            'claheClipLimit': 2.0,
            'claheTileSize': 8,
            'bilateralD': 9,
            'bilateralSigmaColor': 75,
            'bilateralSigmaSpace': 75,
            'useRedChannel': True,
            'contrastBrightness': 1.0,
            'brightnessOffset': 0,
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

    def extract_cell_pattern(self, cell_img, grid_size=5):
        """Extract 3x3 binary pattern from a cell image using 5x5 subdivision"""
        h, w = cell_img.shape
        sub_h, sub_w = h // grid_size, w // grid_size
        
        # Extract center 3x3 pattern from the 5x5 grid
        pattern = []
        for pattern_row in range(3):
            row_pattern = []
            for pattern_col in range(3):
                # Map 3x3 pattern to center of 5x5 grid (indices 1, 2, 3)
                sub_row = pattern_row + 1
                sub_col = pattern_col + 1
                
                # Calculate sub-cell boundaries
                start_y = sub_row * sub_h
                end_y = start_y + sub_h
                start_x = sub_col * sub_w
                end_x = start_x + sub_w
                
                # Extract sub-cell region with padding
                sub_cell = cell_img[start_y+10:end_y-10, start_x+10:end_x-10]
                
                # Calculate average gray value
                avg_gray = np.mean(sub_cell)
                
                binary_value = 1 if avg_gray < 126 else 0
                row_pattern.append(binary_value)
            
            pattern.append(row_pattern)
        
        return pattern
    
    def compare_patterns(self, pattern1, pattern2):
        """Compare two 3x3 patterns and return similarity score (0-1)"""
        if len(pattern1) != 3 or len(pattern2) != 3:
            return 0
        
        matches = 0
        total = 9
        
        for row in range(3):
            for col in range(3):
                if pattern1[row][col] == pattern2[row][col]:
                    matches += 1
        
        return matches / total
    
    def find_best_character_match(self, extracted_pattern, threshold=0.7):
        """Find the best matching character for an extracted pattern"""
        patterns_dict = get_patterns()
        
        best_match = None
        best_score = 0
        
        for char, pattern in patterns_dict.items():
            score = self.compare_patterns(extracted_pattern, pattern)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = char
        
        return best_match, best_score


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
    

    def preprocess_cell_image(self, img):
        """Preprocess a single cell image for better marker detection"""
        use_red = self.params['useRedChannel']
        blur_kernel = 2
        clahe_clip = 2
        clahe_tile = 2
        contrast = 1.0
        brightness = self.params['brightnessOffset']
        
        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 1:
            blur_kernel = 1
        
        # Ensure tile size is positive
        if clahe_tile < 1:
            clahe_tile = 1
        
        # Extract channel
        if use_red:
            processed = img[:, :, 2]  # Red channel (BGR format)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast and brightness
        if contrast != 1.0 or brightness != 0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        
        # Apply Gaussian blur
        if blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # Apply CLAHE (local contrast enhancement)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
            processed = clahe.apply(processed)
        
        return processed
        
        

    def preprocess_image(self, img):
        """Preprocess image - specialized for clay/silica challenges"""
        use_red = self.params['useRedChannel']
        blur_kernel = self.params['gaussianBlurKernel']
        clahe_clip = self.params['claheClipLimit']
        clahe_tile = self.params['claheTileSize']
        bilateral_d = self.params['bilateralD']
        bilateral_color = self.params['bilateralSigmaColor']
        bilateral_space = self.params['bilateralSigmaSpace']
        contrast = self.params['contrastBrightness']
        brightness = self.params['brightnessOffset']
        
        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 1:
            blur_kernel = 1
        
        # Ensure tile size is positive
        if clahe_tile < 1:
            clahe_tile = 1
        
        # Extract channel
        if use_red:
            processed = img[:, :, 2]  # Red channel (BGR format)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast and brightness
        if contrast != 1.0 or brightness != 0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        
        # Apply Gaussian blur
        if blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # Apply CLAHE (local contrast enhancement)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
            processed = clahe.apply(processed)
        
        # Apply bilateral filter
        if bilateral_d > 0:
            processed = cv2.bilateralFilter(processed, bilateral_d, bilateral_color, bilateral_space)
        
        return processed
    
    def detect_markers(self):
        """Detect ArUco markers in the loaded image"""
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
        else:
            print("No markers detected")
        
        return corners, ids, rejected, result_img
    
    def load_templates(self, template_folder='character_templates'):
        """Load all template images from the specified folder"""
        templates = {}
        
        if not os.path.exists(template_folder):
            print(f"Template folder '{template_folder}' not found!")
            return templates
        
        # Support common image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        template_files = []
        
        for extension in image_extensions:
            template_files.extend(glob.glob(os.path.join(template_folder, extension)))
        
        for template_path in template_files:
            # Use filename (without extension) as template name
            template_name = os.path.splitext(os.path.basename(template_path))[0]
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            if template_img is not None:
                templates[template_name] = template_img
                print(f"Loaded template: {template_name}")
            else:
                print(f"Failed to load template: {template_path}")
        
        print(f"Total templates loaded: {len(templates)}")
        return templates
    
    def match_template_in_cell(self, cell_img, template_img, threshold=0.8):
        """Match a template within a cell image"""
        if cell_img.shape[0] < template_img.shape[0] or cell_img.shape[1] < template_img.shape[1]:
            return 0, (0, 0)
        
        # Perform template matching
        result = cv2.matchTemplate(cell_img, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_val, max_loc
    

    def process_grid_with_templates(self, warped_img, templates, grid_rows=9, grid_cols=9, 
                                  cell_width=39, cell_height=39, gap=30, threshold=0.7, 
                                  export_cells=True, cell_export_folder='char_blur'):
        """Process the warped image grid and match templates"""
        height, width = warped_img.shape[:2]
        
        # Calculate grid parameters
        total_grid_width = (grid_cols * cell_width) + ((grid_cols - 1) * gap)
        total_grid_height = (grid_rows * cell_height) + ((grid_rows - 1) * gap)
        
        start_x = (width - total_grid_width) // 2
        start_y = (height - total_grid_height) // 2
        
        # Create export folder if needed
        if export_cells:
            if not os.path.exists(cell_export_folder):
                os.makedirs(cell_export_folder)
                print(f"Created export folder: {cell_export_folder}")
            else:
                # Clear existing files
                for f in glob.glob(os.path.join(cell_export_folder, '*.png')):
                    os.remove(f)
                print(f"Cleared existing files in: {cell_export_folder}")

        
        # Create result visualization
        result_img = warped_img.copy()
        
        # Grid to store results
        grid_results = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        print("\nTemplate matching results:")
        print("-" * 50)
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell position
                cell_x = start_x + col * (cell_width + gap)
                cell_y = start_y + row * (cell_height + gap)

                
                # Extract cell region
                cell_img = warped_img[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width]
                
                cell_img = self.preprocess_cell_image(cell_img)

                result_cell = cell_img.copy()
                # convert to color for visualization
                result_cell = cv2.cvtColor(result_cell, cv2.COLOR_GRAY2BGR)
                #result_img[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width] = result_cell

                # Resize to 500x500
                cell_img = cv2.resize(cell_img, (500, 500), interpolation=cv2.INTER_AREA)

                # Divide cell into 5x5 grid and apply average gray values
                h, w = cell_img.shape
                sub_h, sub_w = h // 5, w // 5

                # draw grid liens on result cell
                #for i in range(1, 5):
                #    cv2.line(result_cell, (0, i * cell_height // 5), (cell_width, i * cell_height // 5), (255, 0, 0), 1)
                #    cv2.line(result_cell, (i * cell_width // 5, 0), (i * cell_width // 5, h), (255, 0, 0), 1)
                
                for sub_row in range(5):
                    for sub_col in range(5):
                        # Calculate sub-cell boundaries
                        _start_y = sub_row * sub_h
                        _end_y = _start_y + sub_h
                        _start_x = sub_col * sub_w
                        _end_x = _start_x + sub_w
                        
                        # Extract sub-cell region
                        sub_cell = cell_img[_start_y+10:_end_y-10, _start_x+10:_end_x-10]

                        print(sub_cell.shape)

                        
                        # Calculate average gray value
                        avg_gray = np.mean(sub_cell)
                        
                        # threshold the value to be either black or white
                        #avg_gray = 255 if avg_gray > 90 else 0

                        # Fill the sub-cell with the average value
                        #cell_img[_start_y:_end_y, _start_x:_end_x] = int(avg_gray)
                
                result_img[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width] = result_cell

                pattern = self.extract_cell_pattern(cell_img, grid_size=5)
                
                # Find best matching template
                best_match = None
                best_score = 0
                best_pos = (0, 0)
                
                for template_name, template_img in templates.items():
                    score, pos = self.match_template_in_cell(cell_img, template_img, threshold)
                    
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = template_name
                        best_pos = pos
                
                # Store result
                if best_match:
                    grid_results[row][col] = {
                        'character': best_match,
                        'confidence': best_score,
                        'position': (cell_x, cell_y)
                    }
                    
                    # Draw result on image
                    cv2.rectangle(result_img, (cell_x, cell_y), 
                                (cell_x + cell_width, cell_y + cell_height), (0, 255, 0), 2)
                    cv2.putText(result_img, f"{best_match}", (cell_x + 2, cell_y + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(result_img, f"{best_score:.2f}", (cell_x + 2, cell_y + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    print(f"Cell ({row},{col}): {best_match} (confidence: {best_score:.3f})")
                else:
                    # Draw empty cell
                    cv2.rectangle(result_img, (cell_x, cell_y), 
                                (cell_x + cell_width, cell_y + cell_height), (0, 0, 255), 1)
        
        if export_cells:
            print(f"\nExported {grid_rows * grid_cols} cell images to '{cell_export_folder}' folder")
        
        return grid_results, result_img


def main():
    # Initialize detector
    detector = ArucoDetector('azulejo5.jpg')
    
    # Detect markers
    print("Detecting ArUco markers...")
    corners, ids, rejected, result = detector.detect_markers()

    # Map marker IDs to positions
    positions = {}
    id_to_position = {11: 'top-right', 10: 'top-left', 13: 'bottom-right', 12: 'bottom-left'}
    corner_mapping = {
        'top-right': 1, 'top-left': 0, 'bottom-right': 2, 'bottom-left': 3
    }

    if ids is not None:
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]
            marker_id_val = int(marker_id[0])
            
            if marker_id_val in id_to_position:
                position_name = id_to_position[marker_id_val]
                corner_index = corner_mapping[position_name]
                corner_point = marker_corners[corner_index]
                
                positions[position_name] = {
                    'x': int(corner_point[0]), 
                    'y': int(corner_point[1]),
                    'corners': marker_corners,
                    'width': int(np.linalg.norm(marker_corners[0] - marker_corners[1])),
                }

        if len(positions) == 4:
            # Apply padding
            padding_config = {
                'top-left': {'x': -16, 'y': -18},
                'top-right': {'x': 14, 'y': -18},
                'bottom-right': {'x': 16, 'y': 20},
                'bottom-left': {'x': -16, 'y': 20}
            }
            
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
            
            # Create corner points for transformation
            tl = (padded_corners['top-left']['x'], padded_corners['top-left']['y'])
            tr = (padded_corners['top-right']['x'], padded_corners['top-right']['y'])
            br = (padded_corners['bottom-right']['x'], padded_corners['bottom-right']['y'])
            bl = (padded_corners['bottom-left']['x'], padded_corners['bottom-left']['y'])
            
            # Perspective transform
            src_pts = np.array([
                [tl[0], tl[1]], [tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]]
            ], dtype=np.float32)
            
            width = height = 600
            dst_pts = np.array([
                [0, 0], [width, 0], [width, height], [0, height]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(result, matrix, (width, height))


            # Load templates and process grid
            print("\nLoading character templates...")
            templates = detector.load_templates('character_templates')
            
            if templates:
                print("\nProcessing grid with template matching...")
                grid_results, result_with_matches = detector.process_grid_with_templates(
                    warped, templates, threshold=0.1
                )
                
                # Display results
                cv2.imshow("Original with Markers", result)
                cv2.imshow("Warped Perspective", warped)
                cv2.imshow("Template Matching Results", result_with_matches)
                
                print("\nGrid processing complete!")
                print("Press any key to close windows...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No templates loaded. Please check the 'character_templates' folder.")
                cv2.imshow("Warped Perspective", warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            cv2.imshow("Original with Markers", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"Not all markers detected. Found {len(positions)} out of 4 required markers.")
    else:
        print("No markers detected!")

if __name__ == "__main__":
    main()