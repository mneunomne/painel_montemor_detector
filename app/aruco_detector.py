"""
Refactored ArUco detector for character recognition on clay surfaces
with frame accumulation for better marker detection
"""
import time
import cv2
import numpy as np
import os
import glob
from character_recognition import recognize_character_from_image, join_characters_to_message
# random
from random import randint as rand
#from cv2 import delay, imshow, waitKey, destroyAllWindows

class ArucoDetector:
    def __init__(self, image_path='image.jpg'):
        """ArUco detector with optimized parameters for red clay surfaces"""
        self.image_path = image_path
        self.original_image = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detection_params = self._get_detection_parameters()
        
        # if path is an URL, read from camera stream
        if self.image_path.startswith('http://') or self.image_path.startswith('https://'):
            # Don't read immediately for URLs - we'll read multiple frames later
            self.is_stream = True
        else:
            self.is_stream = False
            self.load_image()
            
    def _get_detection_parameters(self):
        """Get optimized ArUco detection parameters"""
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshConstant = 7
        params.adaptiveThreshWinSizeMin = 4
        params.adaptiveThreshWinSizeMax = 20
        params.adaptiveThreshWinSizeStep = 8
        
        # Contour filtering parameters
        params.minMarkerPerimeterRate = 0.1  # Lower for small markers
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05  # More lenient for engraved markers
        params.minCornerDistanceRate = 0.05
        params.minMarkerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        
        # Bits extraction parameters - crucial for engraved markers
        params.markerBorderBits = 1
        params.minOtsuStdDev = 3.0  # Lower threshold for uniform surfaces
        params.perspectiveRemovePixelPerCell = 16  # Higher for better bit extraction
        params.perspectiveRemoveIgnoredMarginPerCell = 0.1
        
        # Marker identification parameters - more lenient for damaged engravings
        params.maxErroneousBitsInBorderRate = 1.5  # Allow more border errors
        params.errorCorrectionRate = 1.2  # More aggressive error correction
        
        # Corner refinement parameters - essential for precise detection
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 10
        params.relativeCornerRefinmentWinSize = 0.3
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.05
        #params.adaptiveThreshWinSizeMin = 3
        #params.adaptiveThreshWinSizeMax = 40
        #params.adaptiveThreshWinSizeStep = 10
        #params.minMarkerPerimeterRate = 0.06
        #params.maxMarkerPerimeterRate = 4.0
        #params.polygonalApproxAccuracyRate = 0.03
        #params.minCornerDistanceRate = 0.1
        #params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        #params.cornerRefinementWinSize = 5
        #params.errorCorrectionRate = 0.6
        #params.adaptiveThreshConstant = 0
         
        # params.adaptiveThreshConstant = 0
        # params.minMarkerPerimeterRate = 0.05
        # params.maxMarkerPerimeterRate = 4.0
        # params.polygonalApproxAccuracyRate = 0.03
        # params.minCornerDistanceRate = 0.10
        # params.minDistanceToBorder = 3
        #params.minMarkerDistanceRate = 0.05
        #params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return params
    
    def read_from_camera_stream(self, url):
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # rotate image 180
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame
        else:
            return None
    
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
    
    def preprocess_image(self, frame):
        """Preprocess image for better detection"""
        # Use red channel for clay surfaces
        #processed = img[:, :, 2]  # Red channel (BGR format)
        #processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Apply slight Gaussian blur to reduce noise
        #processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # Apply CLAHE for local contrast enhancement
       # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        #processed = clahe.apply(processed)

        #processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=1.0)
        
        # Apply bilateral filter to reduce noise while preserving edges
        # processed = cv2.bilateralFilter(processed, 9, 75, 75)

        """Enhanced preprocessing for red ceramic surfaces"""
        # Convert to different color spaces for better contrast
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # only use the red channel
        red_channel = frame[:, :, 2]
        
        # Use L channel from LAB for better luminance separation
        gray = red_channel
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply slight Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray
        
        return processed
    
    def preprocess_cell_image(self, img, use_red=True, blur_kernel=2, clahe_clip=2.0, 
                             clahe_tile=8, contrast=1.0, brightness=0):
        """Preprocess a single cell image for better marker detection"""
        
        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 1:
            blur_kernel = 1
        
        # Ensure tile size is positive
        if clahe_tile < 1:
            clahe_tile = 1
        
        # Extract channel
        if use_red and len(img.shape) == 3:
            processed = img[:, :, 2]  # Red channel (BGR format)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
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
    
    def detect_markers(self):
        """Detect ArUco markers in the image"""
        if self.original_image is None:
            print("No image loaded!")
            return None, None, None
        
        # Preprocess and detect
        processed = self.preprocess_image(self.original_image)
        #processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed, self.aruco_dict, parameters=self.detection_params
        )
        
        # Create result visualization
        result_img = self.original_image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(processed, corners, ids)
            print(f"Detected {len(ids)} markers: {ids.flatten()}")
        else:
            print("No markers detected")
        
        return corners, ids, processed
    
    def filter_markers_by_area(self, corners, ids, min_area=500, max_area=5000):
        if ids is None or len(ids) == 0:
            return corners, ids
        
        filtered_corners = []
        filtered_ids = []
        
        for i, corner_set in enumerate(corners):
            # Calculate area using the Shoelace formula
            corner_points = corner_set[0]  # Get the 4 corner points
            area = cv2.contourArea(corner_points)

            print(f"Marker ID: {ids[i][0]}, Area: {area:.2f}")
            
            # Apply filters
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            filtered_corners.append(corner_set)
            filtered_ids.append(ids[i])
        
        return filtered_corners, np.array(filtered_ids) if filtered_ids else None

    def detect_markers_with_accumulation(self, max_attempts=100):
        """
        Detect ArUco markers over multiple frames, accumulating detections
        until all required markers are found or max attempts reached.
        """
        if not self.is_stream:
            # For static images, just do normal detection
            return self.detect_markers()
        
        # Initialize accumulator for marker detections
        accumulated_markers = {}  # {marker_id: {'corners': corners, 'count': count}}
        best_frame = None
        attempt = 0
        
        # Open video stream
        cap = cv2.VideoCapture(self.image_path)

        if not cap.isOpened():
            print(f"Could not open video stream: {self.image_path}")
            return None, None, None
        
        print(f"Attempting to detect markers over {max_attempts} frames...")
        
        while attempt < max_attempts:
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ret:
                print(f"Failed to read frame at attempt {attempt}")
                attempt += 1
                continue
            
            # Resize if too large
            height, width = frame.shape[:2]
            if width > 1200 or height > 900:
                scale = min(800/width, 600/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Store the latest frame
            self.original_image = frame
            best_frame = frame.copy()
            
            # Preprocess and detect
            processed = self.preprocess_image(frame)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            corners, ids, rejected = cv2.aruco.detectMarkers(
                processed_bgr, self.aruco_dict, parameters=self.detection_params
            )
            
            # Accumulate detections
            if ids is not None:
                for i, marker_id in enumerate(ids):
                    marker_id_val = int(marker_id[0])
                    
                    if marker_id_val not in accumulated_markers:
                        accumulated_markers[marker_id_val] = {
                            'corners': corners[i],
                            'count': 1,
                            'last_seen': attempt
                        }
                        print(f"Frame {attempt}: Found new marker {marker_id_val}")
                    else:
                        # Update with latest detection
                        accumulated_markers[marker_id_val]['corners'] = corners[i]
                        accumulated_markers[marker_id_val]['count'] += 1
                        accumulated_markers[marker_id_val]['last_seen'] = attempt
            
            # Check if we have 4 distinct markers
            num_found_markers = len(accumulated_markers)
            
            if num_found_markers >= 4:
                print(f"Found 4 distinct markers after {attempt + 1} attempts!")
                break
            
            # Show progress every 10 frames
            if attempt % 10 == 0:
                found_marker_ids = sorted(accumulated_markers.keys())
                print(f"Attempt {attempt}: Found {num_found_markers}/4 markers. IDs: {found_marker_ids}")
            
            attempt += 1
        
        cap.release()
        
        # Prepare final results
        if len(accumulated_markers) == 0:
            print("No markers detected after all attempts!")
            return None, None, None
        
        # Convert accumulated markers to standard format
        final_corners = []
        final_ids = []
        
        for marker_id, data in accumulated_markers.items():
            final_corners.append(data['corners'])
            final_ids.append([marker_id])
        
        final_ids = np.array(final_ids)
        
        # Create result visualization
        result_img = best_frame.copy() if best_frame is not None else self.original_image.copy()
        cv2.aruco.drawDetectedMarkers(processed_bgr, final_corners, final_ids)
        
        print(f"\nFinal detection summary:")
        print(f"Total attempts: {attempt}")
        print(f"Markers found: {sorted(accumulated_markers.keys())}")
        for marker_id, data in accumulated_markers.items():
            print(f"  Marker {marker_id}: seen {data['count']} times, last at frame {data['last_seen']}")
        
        return final_corners, final_ids, processed_bgr
    
    def get_perspective_transform(self, corners, ids, smallest_id):
        """Get perspective transformation matrix from detected markers"""
        if ids is None or len(ids) < 4:
            print("Not enough markers for perspective transform")
            return None, None
        
        id_to_position = {}
        id_to_position[smallest_id] = 'top-left'
        id_to_position[smallest_id + 1] = 'top-right'
        id_to_position[smallest_id + 2] = 'bottom-left'
        id_to_position[smallest_id + 3] = 'bottom-right'
        
        # Map marker IDs to positions
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
        padding = {'top-left': (-5, -4), 'top-right': (5, -4), 
                  'bottom-right': (5,5), 'bottom-left': (-5, 5)}
        
        src_pts = []
        for corner in ['top-left', 'top-right', 'bottom-right', 'bottom-left']:
            x = positions[corner]['x'] + padding[corner][0]
            y = positions[corner]['y'] + padding[corner][1]
            src_pts.append([x, y])
        
        src_pts = np.array(src_pts, dtype=np.float32)
        
        width = height = 600
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix, (width, height), src_pts
    
    def load_templates(self, template_folder='app/blurred'):
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
                
                # preprocess cell image
                cell_img = self.preprocess_cell_image(cell_img, use_red=True, blur_kernel=3, 
                                                      clahe_clip=2.0, clahe_tile=2, contrast=1.0, brightness=0)
                
                # create cv2 display window
                # cv2.namedWindow(f"Cell ({row},{col})", cv2.WINDOW_NORMAL)
                # pisition the randomly
                # cv2.setWindowProperty(f"Cell ({row},{col})", cv2.WND_PROP_TOPMOST, 1)
                # cv2.moveWindow(f"Cell ({row},{col})", rand(0, 1920), rand(0, 800))
                # Show cell image 
                # cv2.imshow(f"Cell ({row},{col})", cell_img)
                #delay(0.1)  # Allow time to view
                
                
                result_cell = cell_img.copy()
                # convert to color for visualization
                result_cell = cv2.cvtColor(result_cell, cv2.COLOR_GRAY2BGR)

                result_img[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width] = result_cell

                # Export cell if requested
                if export_cells:
                    cell_filename = os.path.join(export_folder, f"cell_{row}_{col}.png")
                    cv2.imwrite(cell_filename, cell_img)
                
                # Recognize character
                character, confidence = recognize_character_from_image(cell_img, row, col, templates)
                
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