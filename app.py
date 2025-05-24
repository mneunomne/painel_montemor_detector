import cv2
import numpy as np
from PIL import Image

def detect_aruco_markers(image_path):
    pil_image = Image.open(image_path)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    positions = {}
    id_to_position = {0: 'left', 1: 'top', 2: 'right', 3: 'bottom'}
    
    if ids is not None:
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]
            center = (int(np.mean(marker_corners[:, 0])), int(np.mean(marker_corners[:, 1])))
            
            marker_id_val = int(marker_id[0])
            if marker_id_val in id_to_position:
                positions[id_to_position[marker_id_val]] = {'x': center[0], 'y': center[1]}
            
            cv2.aruco.drawDetectedMarkers(image, [corners[i]], np.array([[marker_id]]))
            cv2.circle(image, center, 5, (0, 255, 0), -1)

    if len(positions) == 4:
        centers = [(pos['x'], pos['y']) for pos in positions.values()]
        avg_center = (int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers])))
        cv2.circle(image, avg_center, 5, (255, 0, 0), -1)
        print(f"Average Center: {avg_center}")

        # make polygon connecting the centers
        pts = np.array(centers, dtype=np.int32)
        # reorder the points to form a rectangle
        pts = pts[np.argsort([np.arctan2(p[1] - avg_center[1], p[0] - avg_center[0]) for p in centers])]
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # scale up the polygon while keeping it with the same center and orientation
        scale_factor = 2
        center_x, center_y = avg_center
        scaled_pts = []
        for pt in pts:
            scaled_x = int(center_x + (pt[0] - center_x) * scale_factor)
            scaled_y = int(center_y + (pt[1] - center_y) * scale_factor)
            scaled_pts.append((scaled_x, scaled_y))
        scaled_pts = np.array(scaled_pts, dtype=np.int32)
        cv2.polylines(image, [scaled_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Extract ROI using perspective warp
        # Order points: top-left, top-right, bottom-right, bottom-left
        src_pts = np.array([
            (positions['left']['x'], positions['left']['y']),
            (positions['top']['x'], positions['top']['y']),
            (positions['right']['x'], positions['right']['y']),
            (positions['bottom']['x'], positions['bottom']['y'])
        ], dtype=np.float32)
        
        # Define destination rectangle (square output)
        width = height = 400
        dst_pts = np.array([
            [0, height],      # bottom-left
            [0, 0],           # top-left  
            [width, 0],       # top-right
            [width, height]   # bottom-right
        ], dtype=np.float32)
        
        # Get perspective transform matrix and apply warp
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # draw a horizontal line
        # cv2.line(warped, (0, height // 2 - 8), (width, height // 2 - 8), (255, 0, 0), 2)
        # cv2.line(warped, (0, height // 2 + 8), (width, height // 2 + 8), (255, 0, 0), 2)

        # segment width
        segment_width = 16
        
        # draw a vertical line
        # cv2.line(warped, (width // 2 - segment_width // 2, 0), (width // 2 - segment_width // 2, height), (255, 0, 0), 2)
        # cv2.line(warped, (width // 2 + segment_width // 2, 0), (width // 2 + segment_width // 2, height), (255, 0, 0), 2)

        # for segment_width in width
        for i in range(0, width, segment_width):
            cv2.line(warped, (i, 0), (i, height), (255, 0, 0), 1)
        # for segment_width in height
        for i in range(0, height, segment_width):
            cv2.line(warped, (0, i), (width, i), (255, 0, 0), 1)

        

        # Save warped ROI
        cv2.imwrite("roi_warped.png", warped)
        print(f"Warped ROI extracted and saved as roi_warped.png")

    print(f"Positions: {positions}")
    cv2.imwrite("detected_markers.png", image)
    return positions

if __name__ == "__main__":
    detect_aruco_markers("image2.jpg")