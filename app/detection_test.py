import cv2
import numpy as np
import json
import os

class ArucoDetector:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detection_params = self._get_detection_parameters()
        
        # Try to load saved parameters
        # self.load_params_from_file()
        
        # Create only the main display window
        cv2.namedWindow('ArUco Detection', cv2.WINDOW_AUTOSIZE)
        
    def _get_detection_parameters(self):
        params = cv2.aruco.DetectorParameters()
        
        # Optimized parameters for engraved markers on red ceramics
        # Thresholding parameters
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
        
        return params

    def print_current_params(self):
        """Helper function to print current parameter values for debugging"""
        print("=== Current ArUco Detection Parameters ===")
        print(f"Adaptive Threshold Constant: {self.detection_params.adaptiveThreshConstant}")
        print(f"Window Size: {self.detection_params.adaptiveThreshWinSizeMin}-{self.detection_params.adaptiveThreshWinSizeMax} (step: {self.detection_params.adaptiveThreshWinSizeStep})")
        print(f"Marker Perimeter Rate: {self.detection_params.minMarkerPerimeterRate:.3f}-{self.detection_params.maxMarkerPerimeterRate:.3f}")
        print(f"Polygonal Approx Accuracy: {self.detection_params.polygonalApproxAccuracyRate:.3f}")
        print(f"Corner Distance Rate: {self.detection_params.minCornerDistanceRate:.3f}")
        print(f"Marker Distance Rate: {self.detection_params.minMarkerDistanceRate:.3f}")
        print(f"Distance to Border: {self.detection_params.minDistanceToBorder}")
        print(f"Marker Border Bits: {self.detection_params.markerBorderBits}")
        print(f"Min Otsu Std Dev: {self.detection_params.minOtsuStdDev:.1f}")
        print(f"Perspective Pixels/Cell: {self.detection_params.perspectiveRemovePixelPerCell}")
        print(f"Perspective Margin: {self.detection_params.perspectiveRemoveIgnoredMarginPerCell:.3f}")
        print(f"Max Border Error Rate: {self.detection_params.maxErroneousBitsInBorderRate:.3f}")
        print(f"Error Correction Rate: {self.detection_params.errorCorrectionRate:.3f}")
        print(f"Corner Refinement Method: {self.detection_params.cornerRefinementMethod}")
        print(f"Corner Win Size: {self.detection_params.cornerRefinementWinSize}")
        print(f"Relative Corner Win Size: {self.detection_params.relativeCornerRefinmentWinSize:.3f}")
        print(f"Corner Max Iterations: {self.detection_params.cornerRefinementMaxIterations}")
        print(f"Corner Min Accuracy: {self.detection_params.cornerRefinementMinAccuracy:.3f}")
        print("==========================================")

    def save_params_to_file(self, filename="aruco_params.json"):
        """Save current parameters to a JSON file"""
        params_dict = {
            'adaptiveThreshConstant': int(self.detection_params.adaptiveThreshConstant),
            'adaptiveThreshWinSizeMin': self.detection_params.adaptiveThreshWinSizeMin,
            'adaptiveThreshWinSizeMax': self.detection_params.adaptiveThreshWinSizeMax,
            'adaptiveThreshWinSizeStep': self.detection_params.adaptiveThreshWinSizeStep,
            'minMarkerPerimeterRate': self.detection_params.minMarkerPerimeterRate,
            'maxMarkerPerimeterRate': self.detection_params.maxMarkerPerimeterRate,
            'polygonalApproxAccuracyRate': self.detection_params.polygonalApproxAccuracyRate,
            'minCornerDistanceRate': self.detection_params.minCornerDistanceRate,
            'minMarkerDistanceRate': self.detection_params.minMarkerDistanceRate,
            'minDistanceToBorder': self.detection_params.minDistanceToBorder,
            'markerBorderBits': self.detection_params.markerBorderBits,
            'minOtsuStdDev': self.detection_params.minOtsuStdDev,
            'perspectiveRemovePixelPerCell': self.detection_params.perspectiveRemovePixelPerCell,
            'perspectiveRemoveIgnoredMarginPerCell': self.detection_params.perspectiveRemoveIgnoredMarginPerCell,
            'maxErroneousBitsInBorderRate': self.detection_params.maxErroneousBitsInBorderRate,
            'errorCorrectionRate': self.detection_params.errorCorrectionRate,
            'cornerRefinementMethod': self.detection_params.cornerRefinementMethod,
            'cornerRefinementWinSize': self.detection_params.cornerRefinementWinSize,
            'relativeCornerRefinmentWinSize': self.detection_params.relativeCornerRefinmentWinSize,
            'cornerRefinementMaxIterations': self.detection_params.cornerRefinementMaxIterations,
            'cornerRefinementMinAccuracy': self.detection_params.cornerRefinementMinAccuracy
        }
        
        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=4)
        print(f"Parameters saved to {filename}")

    def load_params_from_file(self, filename="aruco_params.json"):
        """Load parameters from a JSON file"""
        if not os.path.exists(filename):
            print(f"Parameter file {filename} not found, using defaults")
            return
        
        try:
            with open(filename, 'r') as f:
                params_dict = json.load(f)
            
            # Apply loaded parameters
            for param_name, value in params_dict.items():
                if hasattr(self.detection_params, param_name):
                    setattr(self.detection_params, param_name, value)
            
            print(f"Parameters loaded from {filename}")
        except Exception as e:
            print(f"Error loading parameters from {filename}: {e}")

    def set_preset_ceramic(self):
        """Preset optimized for engraved markers on ceramics"""
        params = self.detection_params
        params.adaptiveThreshConstant = 5
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.06
        params.maxErroneousBitsInBorderRate = 0.6
        params.errorCorrectionRate = 0.9
        params.perspectiveRemovePixelPerCell = 10
        print("Applied ceramic preset parameters")

    def set_preset_high_contrast(self):
        """Preset for high contrast printed markers"""
        params = self.detection_params
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.05
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.maxErroneousBitsInBorderRate = 0.35
        params.errorCorrectionRate = 0.6
        params.perspectiveRemovePixelPerCell = 4
        print("Applied high contrast preset parameters")

    def set_preset_damaged(self):
        """Preset for damaged or worn markers"""
        params = self.detection_params
        params.adaptiveThreshConstant = 3
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 5.0
        params.polygonalApproxAccuracyRate = 0.08
        params.maxErroneousBitsInBorderRate = 0.7
        params.errorCorrectionRate = 0.95
        params.perspectiveRemovePixelPerCell = 12
        print("Applied damaged marker preset parameters")
    
    def read_from_camera_stream(self, url):
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Resize to 800 width for consistent processing
            frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])), interpolation=cv2.INTER_AREA)
            return frame
        else:
            return None
    
    def preprocess_frame(self, frame):
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
    
    def detect_markers(self, frame):
        # Enhanced preprocessing for red ceramics
        gray = self.preprocess_frame(frame)
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detection_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        return corners, ids, rejected, gray
    
    def draw_markers(self, frame, corners, ids):
        if ids is not None:
            # Draw detected markers with thicker lines for visibility
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
            
            # Add detailed marker information
            for i, corner in enumerate(corners):
                # Calculate marker properties
                center = np.mean(corner[0], axis=0).astype(int)
                
                # Calculate marker size (average side length)
                side_lengths = []
                for j in range(4):
                    p1 = corner[0][j]
                    p2 = corner[0][(j+1)%4]
                    side_length = np.linalg.norm(p2 - p1)
                    side_lengths.append(side_length)
                avg_size = np.mean(side_lengths)
                
                # Draw marker ID with background
                id_text = f'ID: {ids[i][0]}'
                text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (center[0]-text_size[0]//2-5, center[1]-30), 
                             (center[0]+text_size[0]//2+5, center[1]-5), (0, 0, 0), -1)
                cv2.putText(frame, id_text, (center[0]-text_size[0]//2, center[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw size information
                size_text = f'Size: {avg_size:.1f}px'
                cv2.putText(frame, size_text, (center[0]-50, center[1]+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw center point
                cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)
                
                # Draw corner numbers
                for j, point in enumerate(corner[0]):
                    pt = point.astype(int)
                    cv2.circle(frame, tuple(pt), 4, (255, 0, 0), -1)
                    cv2.putText(frame, str(j), (pt[0]+5, pt[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def add_info_overlay(self, frame, corners, ids, rejected):
        # Enhanced detection info overlay
        height, width = frame.shape[:2]
        
        # Detection statistics
        detected_count = len(ids) if ids is not None else 0
        rejected_count = len(rejected) if rejected is not None else 0
        
        # Calculate detection rate (simple moving average would be better)
        total_candidates = detected_count + rejected_count
        detection_rate = (detected_count / total_candidates * 100) if total_candidates > 0 else 0
        
        info_text = [
            f"Detected: {detected_count}",
            f"Rejected: {rejected_count}",
            f"Detection Rate: {detection_rate:.1f}%",
            f"Frame Size: {width}x{height}",
            "",
            "Controls:",
            "ESC - Exit",
            "S - Save frame",
            "P - Print params",
            "1 - Ceramic preset",
            "2 - High contrast preset", 
            "3 - Damaged preset",
            "R - Save params"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_height = len(info_text) * 18 + 20
        cv2.rectangle(overlay, (10, 10), (280, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text with better formatting
        for i, text in enumerate(info_text):
            color = (0, 255, 0) if "Controls:" in text or text == "" else (255, 255, 255)
            if "Detected:" in text:
                color = (0, 255, 0)
            elif "Rejected:" in text:
                color = (0, 0, 255) 
            elif "Detection Rate:" in text:
                color = (0, 255, 255)
                
            cv2.putText(frame, text, (15, 30 + i*18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run(self):
        print("Starting Enhanced ArUco Detection...")
        print("=== Keyboard Controls ===")
        print("ESC - Exit")
        print("S - Save current frame")
        print("P - Print current parameters")
        print("R - Save parameters to file")
        print("1 - Apply ceramic preset")
        print("2 - Apply high contrast preset")
        print("3 - Apply damaged marker preset")
        print("=========================")
        
        frame_count = 0
        
        while True:
            # Read frame from stream
            frame = self.read_from_camera_stream(self.stream_url)
            
            if frame is None:
                print("Failed to read frame from stream")
                continue
            
            # Detect markers
            corners, ids, rejected, processed_gray = self.detect_markers(frame)
            
            # Draw markers and info
            frame = self.draw_markers(frame, corners, ids)
            frame = self.add_info_overlay(frame, corners, ids, rejected)
            
            # Display frame
            cv2.imshow('ArUco Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s') or key == ord('S'):
                filename = f'aruco_frame_{frame_count:04d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
            elif key == ord('p') or key == ord('P'):
                self.print_current_params()
            elif key == ord('r') or key == ord('R'):
                self.save_params_to_file()
            elif key == ord('1'):
                self.set_preset_ceramic()
            elif key == ord('2'):
                self.set_preset_high_contrast()
            elif key == ord('3'):
                self.set_preset_damaged()
        
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Replace with your camera stream URL
    stream_url = "http://192.168.31.227:5000/video_feed"
    
    detector = ArucoDetector(stream_url)
    detector.run()