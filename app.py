import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def detect_aruco_markers(image_path):
    """
    Detect ArUco markers in a PNG image and return their IDs and positions
    
    Args:
        image_path (str): Path to the PNG image file
        
    Returns:
        dict: Dictionary containing marker IDs, corners, and centers
    """
    
    # Read the image
    try:
        # Using PIL to handle PNG files properly
        pil_image = Image.open(image_path)
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the ArUco dictionary (using DICT_6X6_250 as default)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    
    results = {
        'ids': [],
        'corners': [],
        'centers': [],
        'image': image.copy()
    }
    
    if ids is not None:
        # Process each detected marker
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]  # Get the four corners
            
            # Calculate center point
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            results['ids'].append(int(marker_id[0]))
            results['corners'].append(marker_corners)
            results['centers'].append((center_x, center_y))
            
            # Draw marker outline and ID on the image
            cv2.aruco.drawDetectedMarkers(results['image'], [corners[i]], np.array([[marker_id]]))
            
            # Draw center point
            cv2.circle(results['image'], (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Add text with marker ID and position
            text = f"ID:{marker_id[0]} ({center_x},{center_y})"
            cv2.putText(results['image'], text, (center_x + 10, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return results

def display_results(results):
    """
    Display the detection results
    """
    if results is None:
        print("No results to display")
        return
    
    print("=== ArUco Marker Detection Results ===")
    
    if len(results['ids']) == 0:
        print("No ArUco markers detected in the image.")
    else:
        print(f"Found {len(results['ids'])} marker(s):")
        print("-" * 40)
        
        for i, marker_id in enumerate(results['ids']):
            center = results['centers'][i]
            corners = results['corners'][i]
            
            print(f"Marker ID: {marker_id}")
            print(f"  Center Position: ({center[0]}, {center[1]})")
            print(f"  Corner Coordinates:")
            for j, corner in enumerate(corners):
                print(f"    Corner {j+1}: ({corner[0]:.1f}, {corner[1]:.1f})")
            print("-" * 40)
    
    # Display the image with detected markers
    plt.figure(figsize=(12, 8))
    # Convert BGR to RGB for matplotlib
    display_image = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)
    plt.imshow(display_image)
    plt.title(f"ArUco Marker Detection - Found {len(results['ids'])} marker(s)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your PNG image path
    image_path = "image.jpg"
    
    print("ArUco Marker Detector")
    print("=" * 30)
    
    # Detect markers
    results = detect_aruco_markers(image_path)
    
    if results:
        # Display results
        display_results(results)
        
        # Optional: Save the annotated image
        output_path = "detected_markers.png"
        cv2.imwrite(output_path, results['image'])
        print(f"\nAnnotated image saved as: {output_path}")
    else:
        print("Failed to process the image.")

# Alternative function to test with different ArUco dictionaries
def detect_with_multiple_dictionaries(image_path):
    """
    Try detecting with multiple ArUco dictionaries
    """
    dictionaries = [
        (cv2.aruco.DICT_4X4_50, "DICT_4X4_50"),
        (cv2.aruco.DICT_4X4_100, "DICT_4X4_100"),
        (cv2.aruco.DICT_4X4_250, "DICT_4X4_250"),
        (cv2.aruco.DICT_5X5_50, "DICT_5X5_50"),
        (cv2.aruco.DICT_5X5_100, "DICT_5X5_100"),
        (cv2.aruco.DICT_5X5_250, "DICT_5X5_250"),
        (cv2.aruco.DICT_6X6_50, "DICT_6X6_50"),
        (cv2.aruco.DICT_6X6_100, "DICT_6X6_100"),
        (cv2.aruco.DICT_6X6_250, "DICT_6X6_250"),
    ]
    
    best_result = None
    max_markers = 0
    
    for dict_type, dict_name in dictionaries:
        # Read image
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try detection with this dictionary

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        markers_found = len(ids) if ids is not None else 0
        print(f"{dict_name}: {markers_found} markers found")
        
        if markers_found > max_markers:
            max_markers = markers_found
            best_result = (corners, ids, dict_name)
    
    return best_result