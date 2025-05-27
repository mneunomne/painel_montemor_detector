import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import re

max_length = 0

def normalize_text(text, max_length=18):
    """
    Normalize text to fit within max_length characters.
    Removes special characters, converts to uppercase, and truncates or pads.
    """
    if pd.isna(text):
        text = "UNKNOWN"
    else:
        text = str(text)
        
    # Convert to uppercase
    text = text.upper().strip()
    
    return text

def normalize_coordinate(coord, coord_type='lat'):
    """
    Normalize coordinates to fit within 18 characters.
    """
    if pd.isna(coord):
        return "0.0".ljust(18)
    
    # Convert to string with reasonable precision
    coord_str = f"{float(coord):.6f}"
    
    return coord_str

def normalize_date(date_str):
    """
    Normalize date to fit within 18 characters.
    """
    
    date_str = str(date_str).strip()
    
    # Normalize the date string
    date_str = re.sub(r'[^A-Za-z0-9\s.,-]', '', date_str)
    date_str = date_str.upper().strip()
    
    return date_str

def get_patterns():
    """
    Return the pattern dictionary used for character encoding.
    """
    return {
    # Basic Latin letters - improved for better recognition
    'A': [[0,1,0], [1,1,1], [1,0,1]],      # Classic A shape with peak
    'B': [[1,1,0], [1,1,1], [1,1,0]],      # B with two bumps (kept good)
    'C': [[0,1,1], [1,0,0], [0,1,1]],      # C opening to right
    'D': [[1,1,0], [1,0,1], [1,1,0]],      # D shape (kept good)
    'E': [[1,1,1], [1,1,0], [1,1,1]],      # E with horizontal lines (kept)
    'F': [[1,1,1], [1,1,0], [1,0,0]],      # F without bottom (kept)
    'G': [[0,1,1], [1,0,0], [1,1,1]],      # G with bottom closing
    'H': [[1,0,1], [1,1,1], [1,0,1]],      # Perfect H (kept)
    'I': [[1,1,1], [0,1,0], [1,1,1]],      # I with serifs (kept)
    'J': [[0,0,1], [0,0,1], [1,1,1]],      # J hook at bottom
    'K': [[1,0,1], [1,1,0], [1,0,1]],      # K with angles (kept)
    'L': [[1,0,0], [1,0,0], [1,1,1]],      # L shape (kept)
    'M': [[1,0,1], [1,1,1], [1,0,1]],      # M with center connection
    'N': [[1,0,1], [1,1,1], [1,0,1]],      # N with diagonal (similar to M but contextual)
    'O': [[0,1,0], [1,0,1], [0,1,0]],      # Better circular O
    'P': [[1,1,1], [1,1,0], [1,0,0]],      # P shape (kept)
    'Q': [[0,1,0], [1,0,1], [0,1,1]],      # Q with distinctive tail
    'R': [[1,1,0], [1,1,1], [1,0,1]],      # R with leg (kept)
    'S': [[0,1,1], [0,1,0], [1,1,0]],      # S curve (kept)
    'T': [[1,1,1], [0,1,0], [0,1,0]],      # T shape (kept)
    'U': [[1,0,1], [1,0,1], [0,1,0]],      # U with curved bottom
    'V': [[1,0,1], [1,0,1], [0,1,0]],      # V converging (kept)
    'W': [[1,0,1], [1,0,1], [1,1,1]],      # W with wide base (kept)
    'X': [[1,0,1], [0,1,0], [1,0,1]],      # X cross (kept)
    'Y': [[1,0,1], [0,1,0], [0,1,0]],      # Y shape (kept)
    'Z': [[1,1,1], [0,1,0], [1,1,1]],      # Z with diagonal (kept)
    
    # Numbers - more distinctive and recognizable
    '0': [[0,1,0], [1,0,1], [0,1,0]],      # Oval zero
    '1': [[0,1,0], [0,1,0], [0,1,0]],      # Simple vertical line
    '2': [[1,1,0], [0,1,0], [1,1,1]],      # 2 with curves
    '3': [[1,1,0], [0,1,0], [1,1,0]],      # 3 with bumps on right
    '4': [[1,0,1], [1,1,1], [0,0,1]],      # 4 shape (kept)
    '5': [[1,1,1], [1,1,0], [1,1,0]],      # 5 with top and middle
    '6': [[0,1,0], [1,1,0], [0,1,0]],      # 6 with loop
    '7': [[1,1,1], [0,0,1], [0,0,1]],      # 7 shape (kept)
    '8': [[0,1,0], [1,1,1], [0,1,0]],      # 8 with middle connection
    '9': [[0,1,0], [0,1,1], [0,1,0]],      # 9 with top loop
    
    # Punctuation and special characters
    ' ': [[0,0,0], [0,0,0], [0,0,0]],      # Empty space (kept)
    '.': [[0,0,0], [0,0,0], [0,1,0]],      # Period (kept)
    ';': [[0,1,0], [0,0,0], [0,1,1]],      # Semicolon (improved)
    '-': [[0,0,0], [1,1,1], [0,0,0]],      # Hyphen (kept)
    '|': [[0,1,0], [0,1,0], [0,1,0]],      # Vertical bar (improved)
    '/': [[0,0,1], [0,1,0], [1,0,0]],      # Forward slash
    }

def create_character_image(char, cell_size=20):
    """
    Create a pattern image for a single character using its 3x3 pattern.
    """
    patterns = get_patterns()
    
    # Create 5x5 image (3x3 pattern with 2 cell border border)
    img_width = 4 * cell_size
    img_height = 4 * cell_size
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Get pattern for character
    if char in patterns:
        pattern = patterns[char]
        
        # Draw 3x3 pattern in the 5x5 grid
        for row in range(3):
            for col in range(3):
                x1 = (col + 0.5) * cell_size
                y1 = (row + 0.5) * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                color = 'black' if pattern[row][col] == 1 else 'white'
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='white')
    
    return img

def create_dictionary_image(output_path="pattern_dictionary.png", img_size=1000):
    """
    Create a 1000x1000 dictionary image showing all patterns and their corresponding characters.
    """
    patterns = get_patterns()
    
    # Create image
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to use a larger font
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Calculate layout
    chars_per_row = 8  # 8 characters per row to fit nicely in 1000px
    cell_size = 15  # Size of each pattern cell
    pattern_width = 3 * cell_size  # 3x3 pattern
    char_width = 30  # Space for character display
    total_item_width = pattern_width + char_width + 20  # 20px spacing
    
    # Calculate starting positions to center the grid
    grid_width = chars_per_row * total_item_width
    start_x = (img_size - grid_width) // 2
    start_y = 50
    
    # Sort characters for better organization
    sorted_chars = sorted(patterns.keys(), key=lambda x: (x.isdigit(), x.isalpha(), x))
    
    # Draw each character and its pattern
    for idx, char in enumerate(sorted_chars):
        pattern = patterns[char]
        
        # Calculate position
        row = idx // chars_per_row
        col = idx % chars_per_row
        
        x_pos = start_x + col * total_item_width
        y_pos = start_y + row * (pattern_width + 80)  # 80px vertical spacing
        
        # Draw the 3x3 pattern
        for pattern_row in range(3):
            for pattern_col in range(3):
                x1 = x_pos + pattern_col * cell_size
                y1 = y_pos + pattern_row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                color = 'black' if pattern[pattern_row][pattern_col] == 1 else 'white'
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray', width=1)
        
        # Draw the character label
        char_x = x_pos + pattern_width + 10
        char_y = y_pos + cell_size  # Center vertically with pattern
        
        if font:
            draw.text((char_x, char_y), char, fill='black', font=font)
        else:
            # Fallback for systems without font support
            draw.text((char_x, char_y), char, fill='black')
    
    # Save the image
    img.save(output_path)
    print(f"Dictionary image saved as '{output_path}'")
    return img

def create_char_templates(output_dir="character_templates", cell_size=100):
    """
    Create individual character template images for each character in the pattern dictionary.
    Each character will have its own image file in the specified output directory.
    """
    patterns = get_patterns()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for char, pattern in patterns.items():
        # Create character image
        img = create_character_image(char, cell_size=cell_size)
        
        # if character is a space or ponctuation, change name
        if char == ' ':
            char = 'SPACE'
        elif char == '.':
            char = 'DOT'
        elif char == ',':
            char = 'COMMA'
        elif char == '-':
            char = 'HYPHEN'
        elif char == '/':
            char = 'BAR'
        elif char == ':':
            char = 'COLON'
        elif char == ';':
            char = 'SEMICOLON'
        elif char == '|':
            char = 'PIPE'

        # Save image
        filename = f"{char}.png"
        img.save(os.path.join(output_dir, filename))
        print(f"Saved template for '{char}' as '{filename}'")

# create full image with data grid with individual character images
def create_image_grid(idx, images, cell_size=15, cols=9, img_width=1000):
    """
    Create a grid image from a list of images.
    Each image will be resized to fit in the grid cells.
    """
    rows = cols  # Calculate number of rows needed

    img_height = img_width
    
    # Create a tranparent blank image
    grid_img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))

    draw = ImageDraw.Draw(grid_img)

    gap = 35  # Gap between images

    margin = 40  # Margin around the grid
    cell_size = (img_width - 2 * margin - (cols - 1) * gap) // cols
    cell_size = min(cell_size, img_height // rows)  # Ensure it fits in height

    image_index = 0
    marker_id = idx  # Starting marker ID

    for row in range(rows):
        for col in range(cols):
            # Calculate position for this grid cell
            x_pos = col * cell_size + margin + col * gap
            y_pos = row * cell_size + margin + row * gap
            
            # Check if this is a corner position (where we skip images)
            is_corner = False
            corner_index = -1
            
            if col == 0 and row == 0:
                is_corner = True
                corner_index = 0
            elif col == cols - 1 and row == rows - 1:
                is_corner = True
                corner_index = 3
            elif col == 0 and row == rows - 1:
                is_corner = True
                corner_index = 2
            elif col == cols - 1 and row == 0:
                is_corner = True
                corner_index = 1
            
            if is_corner:
                # Draw fiducial marker if this is the specific corner cell
                if corner_index >= 0:
                    # White background size spans 2 cells + 1 gap
                    white_bg_size = cell_size
                    
                    # Calculate padding and marker size
                    padding = int(cell_size / 5) // 2
                    marker_size = white_bg_size - (padding * 2)
                    
                    # Create drawing context
                    draw = ImageDraw.Draw(grid_img)
                    
                    # Draw white rectangle background
                    draw.rectangle([
                        x_pos, y_pos, 
                        x_pos + white_bg_size, y_pos + white_bg_size
                    ], fill='white')
                    
                    # Load and resize marker image
                    marker_id_str = f"{marker_id:03d}"
                    marker_img = Image.open(f"aruco_markers/aruco_marker_{marker_id_str}.png")
                    marker_img = marker_img.resize((marker_size, marker_size))
                    
                    # Convert to RGBA if not already
                    if marker_img.mode != 'RGBA':
                        marker_img = marker_img.convert('RGBA')
                    
                    # Calculate marker position (centered within white background)
                    marker_x = x_pos + padding
                    marker_y = y_pos + padding
                    
                    # Paste the marker
                    grid_img.paste(marker_img, (marker_x, marker_y), marker_img)
                    
                    marker_id += 1
                
                continue  # Skip placing images in corner areas
            
            # Place regular image if we have one
            if image_index >= len(images):
                break
                
            print(f"Processing image {image_index}/{len(images)} at grid position ({col}, {row})")
            img = images[image_index]
            
            # Resize image to fit in cell
            img_resized = img.resize((cell_size, cell_size))
            
            # Paste the resized image into the grid
            grid_img.paste(img_resized, (x_pos, y_pos))
            
            image_index += 1

    return grid_img

def process_csv_to_individual_characters(csv_file_path, output_dir="character_patterns"):
    """
    Process CSV file and generate individual character pattern images for each entry.
    Each entry gets its own folder with individual character images.
    """
    max_length = 0
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Try to identify the correct columns
    lat_col = None
    lng_col = None
    name_col = None
    date_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'lat' in col_lower and lat_col is None:
            lat_col = col
        elif 'lng' in col_lower and lng_col is None:
            lng_col = col
        elif 'name' in col_lower and name_col is None:
            name_col = col
        elif 'date' in col_lower and date_col is None:
            date_col = col
        
    print(f"Using: lat={lat_col}, lng={lng_col}, name={name_col}, date={date_col}")
    
    # Process each row
    for idx, row in df.iterrows():
        data_images = []
        print(f"Processing row {idx + 1}/{len(df)}")
        
        # Create entry folder
        row_id = idx + 1  # Use 1-based indexing for IDs
        
        # Extract and normalize data
        lat_data = row[lat_col] if lat_col else None
        lng_data = row[lng_col] if lng_col else None
        name_data = row[name_col] if name_col else None
        date_data = row[date_col] if date_col else None
        
        tile_data = row.get('tile', None)

        
        # Normalize each field to 18 characters
        normalized_lat = normalize_coordinate(lat_data, 'lat')
        normalized_lng = normalize_coordinate(lng_data, 'lng')
        normalized_name = normalize_text(name_data)
        normalized_date = normalize_date(date_data)

        filename = normalized_name
        
        # replace space with underscore and remove special characters
        filename = re.sub(r' ', '_', filename)
        path = os.path.join(output_dir, f"{re.sub(r'[^A-Za-z0-9_]', '', filename).lower()}.png")

        data = f"{normalized_name}|{normalized_date}"
        data = data.ljust(49, ';')        

        max_length = max(max_length, len(data))
        
        print(f"  Normalized data: '{data}'")
        
        # Define field data
        fields = {
            'name': normalized_name,
            'date': normalized_date,
            'lat': normalized_lat,
            'lng': normalized_lng
        }
        
        # Generate image for each character
        for char_idx, char in enumerate(data):
            char_img = create_character_image(char, cell_size=20)
            if char == ';':
                print(f"  Skipping character '{char}' at index {char_idx} (semicolon)")
                # char img is transparent
                char_img = Image.new('RGBA', char_img.size, (255, 255, 255, 0))
            
            data_images.append(char_img)
            
            # Save character image
            #char_img.save(os.path.join(entry_folder, filename))
        
        print(f"  Saved {len(data_images)} character images ")
        # create image grid        
        grid_img = create_image_grid(idx, data_images, cell_size=20, cols=7, img_width=1000)

        grid_img.save(path)        

        # get image in inverted/{tile_data}.png
        tile_path = os.path.join("inverted", f"tile_{tile_data}.png")
        tile_img = Image.open(tile_path)
        tile_img = tile_img.resize((1000, 1000))
        

        # grid image in the tile image
        tile_img.paste(grid_img, (0, 0), grid_img)

        # convert to binary image for laser engraving
        tile_img = tile_img.convert("1")

        tile_img.save(os.path.join("tiles", f"{tile_data}.png"))

        # data_images.append(tile_img)


        print(f"  Created grid image for entry {row_id}")
    return max_length

if __name__ == "__main__":
    # First, create the dictionary image
    print("Creating pattern dictionary image...")
    create_dictionary_image("pattern_dictionary.png", 1000)
    create_char_templates("character_templates", cell_size=100)
    # Then process the CSV file
    csv_file_path = "data.csv"
    
    # Process the CSV and generate individual character pattern images
    max_length = process_csv_to_individual_characters(csv_file_path)

    print(f"max_length: {max_length}")