import cv2
import numpy as np
from PIL import Image

def draw_perspective_grid(image, positions, grid_size=10):
    """
    Draw a grid on the original image that follows the perspective defined by the markers.
    Uses bilinear interpolation to create a grid that respects the diamond orientation.
    """
    # Get marker positions as numpy arrays
    top = np.array([positions['top']['x'], positions['top']['y']], dtype=np.float32)
    right = np.array([positions['right']['x'], positions['right']['y']], dtype=np.float32)
    bottom = np.array([positions['bottom']['x'], positions['bottom']['y']], dtype=np.float32)
    left = np.array([positions['left']['x'], positions['left']['y']], dtype=np.float32)
    
    # Create grid using bilinear interpolation
    # For a diamond shape, we need to interpolate differently
    
    # Draw lines from top-left to bottom-right
    for i in range(grid_size + 1):
        t = i / grid_size
        
        # Lines parallel to top-left to bottom-right diagonal
        start = (1 - t) * left + t * top
        end = (1 - t) * bottom + t * right
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 0), 1)
        
        # Lines parallel to top-right to bottom-left diagonal
        start = (1 - t) * top + t * right
        end = (1 - t) * left + t * bottom
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 0), 1)
    
    # Also draw the main diagonals thicker for reference
    cv2.line(image, tuple(top.astype(int)), tuple(bottom.astype(int)), (0, 255, 255), 2)
    cv2.line(image, tuple(left.astype(int)), tuple(right.astype(int)), (0, 255, 255), 2)

def draw_perspective_grid_quadrilateral(image, positions, grid_size=10):
    """
    Alternative method: Draw a perspective grid by treating the markers as corners of a quadrilateral.
    This creates a grid that would appear square when viewed from above.
    """
    # Define source points (marker positions in diamond configuration)
    src_pts = np.array([
        [positions['top']['x'], positions['top']['y']],
        [positions['right']['x'], positions['right']['y']],
        [positions['bottom']['x'], positions['bottom']['y']],
        [positions['left']['x'], positions['left']['y']]
    ], dtype=np.float32)
    
    # Create a virtual square grid
    grid_pts = []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            grid_pts.append([i / grid_size, j / grid_size])
    grid_pts = np.array(grid_pts, dtype=np.float32)
    
    # For each grid point, calculate its position in the perspective view
    perspective_pts = []
    for pt in grid_pts:
        u, v = pt
        # Bilinear interpolation
        p = (1-u)*(1-v)*src_pts[3] + u*(1-v)*src_pts[0] + u*v*src_pts[1] + (1-u)*v*src_pts[2]
        perspective_pts.append(p)
    
    # Draw horizontal lines
    for i in range(grid_size + 1):
        for j in range(grid_size):
            idx1 = i * (grid_size + 1) + j
            idx2 = i * (grid_size + 1) + j + 1
            cv2.line(image, 
                    tuple(perspective_pts[idx1].astype(int)), 
                    tuple(perspective_pts[idx2].astype(int)), 
                    (255, 255, 0), 1)
    
    # Draw vertical lines
    for i in range(grid_size):
        for j in range(grid_size + 1):
            idx1 = i * (grid_size + 1) + j
            idx2 = (i + 1) * (grid_size + 1) + j
            cv2.line(image, 
                    tuple(perspective_pts[idx1].astype(int)), 
                    tuple(perspective_pts[idx2].astype(int)), 
                    (255, 255, 0), 1)

def detect_aruco_markers(image_path):
    """
    Detect ArUco markers and perform perspective transform to get a top-down view
    of the region defined by the markers.
    """
    # Load image
    pil_image = Image.open(image_path)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Setup ArUco detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Map marker IDs to positions
    positions = {}
    id_to_position = {0: 'left', 1: 'top', 2: 'right', 3: 'bottom'}
    
    if ids is not None:
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]
            center = (int(np.mean(marker_corners[:, 0])), int(np.mean(marker_corners[:, 1])))
            
            marker_id_val = int(marker_id[0])
            if marker_id_val in id_to_position:
                positions[id_to_position[marker_id_val]] = {
                    'x': center[0], 
                    'y': center[1],
                    'corners': marker_corners
                }
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image, [corners[i]], np.array([[marker_id]]))
            cv2.circle(image, center, 5, (0, 255, 0), -1)
            
            # Label the markers
            cv2.putText(image, id_to_position.get(marker_id_val, str(marker_id_val)), 
                       (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(positions) == 4:
        # Calculate the center of all markers
        centers = [(pos['x'], pos['y']) for pos in positions.values()]
        avg_center = (int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers])))
        cv2.circle(image, avg_center, 5, (255, 0, 0), -1)
        print(f"Average Center: {avg_center}")
        
        # Draw polygon connecting the marker centers
        pts = np.array([
            (positions['top']['x'], positions['top']['y']),
            (positions['right']['x'], positions['right']['y']),
            (positions['bottom']['x'], positions['bottom']['y']),
            (positions['left']['x'], positions['left']['y'])
        ], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Create a copy for the second grid style
        image_copy = image.copy()
        
        # Draw diamond-oriented grid lines on the original image
        draw_perspective_grid(image, positions, grid_size=10)
        
        # Draw quadrilateral perspective grid on the copy
        draw_perspective_grid_quadrilateral(image_copy, positions, grid_size=10)
        
        # Show both versions
        cv2.imshow("Diamond Grid Overlay", image)
        cv2.imshow("Perspective Grid Overlay", image_copy)

        # Extract ROI using perspective warp
        # Order points clockwise starting from top-left
        src_pts = np.array([
            (positions['top']['x'], positions['top']['y']),      # top
            (positions['right']['x'], positions['right']['y']),  # right
            (positions['bottom']['x'], positions['bottom']['y']), # bottom
            (positions['left']['x'], positions['left']['y'])     # left
        ], dtype=np.float32)
        
        # Define destination rectangle (square output)
        width = height = 600  # Increased size for better detail
        
        # Map to a square with proper orientation
        dst_pts = np.array([
            [width//2, 0],          # top center
            [width, height//2],     # right center
            [width//2, height],     # bottom center
            [0, height//2]          # left center
        ], dtype=np.float32)
        
        # Get perspective transform matrix and apply warp
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # Add grid overlay
        segment_width = 25  # Grid cell size
        
        # Draw vertical lines
        for i in range(0, width, segment_width):
            cv2.line(warped, (i, 0), (i, height), (128, 128, 128), 1)
        
        # Draw horizontal lines
        for i in range(0, height, segment_width):
            cv2.line(warped, (0, i), (width, i), (128, 128, 128), 1)
        
        # Analyze central segments
        center_y = height // 2
        center_x = width // 2
        
        # Horizontal analysis
        print("\nHorizontal center line analysis:")
        for i in range(0, width, segment_width):
            segment = warped[center_y - segment_width//2:center_y + segment_width//2, 
                            i:i + segment_width]
            if segment.size > 0:
                avg_color = cv2.mean(segment)[:3]
                print(f"  Position {i}: BGR{tuple(map(int, avg_color))}")
                # Draw color indicator
                cv2.rectangle(warped, (i, height-30), (i+segment_width, height), 
                            tuple(map(int, avg_color)), -1)
        
        # Vertical analysis
        print("\nVertical center line analysis:")
        for i in range(0, height, segment_width):
            segment = warped[i:i + segment_width, 
                            center_x - segment_width//2:center_x + segment_width//2]
            if segment.size > 0:
                avg_color = cv2.mean(segment)[:3]
                print(f"  Position {i}: BGR{tuple(map(int, avg_color))}")
                # Draw color indicator
                cv2.rectangle(warped, (width-30, i), (width, i+segment_width), 
                            tuple(map(int, avg_color)), -1)
        
        # Draw center lines
        cv2.line(warped, (0, center_y), (width, center_y), (0, 255, 255), 2)
        cv2.line(warped, (center_x, 0), (center_x, height), (0, 255, 255), 2)
        
        # Display the warped image
        cv2.imshow("Warped Perspective View", warped)
        
        # Also create a cleaner version without annotations
        warped_clean = cv2.warpPerspective(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), 
                                          matrix, (width, height))
        cv2.imshow("Clean Warped View", warped_clean)
        
    else:
        print(f"Not all markers detected. Found {len(positions)} markers:")
        for pos, data in positions.items():
            print(f"  {pos}: ({data['x']}, {data['y']})")
    
    # Display the original image with annotations
    # cv2.imshow("Detected Markers", image) - removed as we show grid versions instead
    
    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return positions, matrix if len(positions) == 4 else None

def get_real_world_coordinates(pixel_point, homography_matrix):
    """
    Convert pixel coordinates to real-world coordinates using the homography matrix.
    """
    # Add homogeneous coordinate
    point = np.array([pixel_point[0], pixel_point[1], 1.0])
    
    # Apply inverse homography
    inv_matrix = np.linalg.inv(homography_matrix)
    world_point = inv_matrix @ point
    
    # Normalize
    world_point = world_point / world_point[2]
    
    return world_point[:2]

if __name__ == "__main__":
    positions, homography = detect_aruco_markers("image3.jpg")
    
    if homography is not None:
        print("\nHomography matrix computed successfully!")
        print("You can now map any point from the warped view back to the original image.")