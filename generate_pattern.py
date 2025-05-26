import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import re

def normalize_text(text, max_length=18):
    """
    Normalize text to fit within max_length characters.
    Removes special characters, converts to uppercase, and truncates or pads.
    """
    if pd.isna(text):
        text = "UNKNOWN"
    else:
        text = str(text)
    
    # Remove special characters except spaces, periods, commas, and hyphens
    text = re.sub(r'[^A-Za-z0-9\s.,-]', '', text)
    
    # Convert to uppercase
    text = text.upper().strip()
    
    # Truncate or pad to exact length
    if len(text) > max_length:
        text = text[:max_length]
    else:
        text = text.ljust(max_length)
    
    return text

def normalize_coordinate(coord, coord_type='lat'):
    """
    Normalize coordinates to fit within 18 characters.
    """
    if pd.isna(coord):
        return "0.0".ljust(18)
    
    # Convert to string with reasonable precision
    coord_str = f"{float(coord):.6f}"
    
    # Add coordinate type prefix if needed
    if coord_type == 'lat':
        prefix = "LAT"
    else:
        prefix = "LNG"
    
    # Format as "LAT-12.345678" or similar
    formatted = f"{prefix}{coord_str}"
    
    # Truncate or pad to 18 characters
    if len(formatted) > 18:
        formatted = formatted[:18]
    else:
        formatted = formatted.ljust(18)
    
    return formatted

def normalize_date(date_str):
    """
    Normalize date to fit within 18 characters.
    """
    if pd.isna(date_str):
        return "UNKNOWN DATE".ljust(18)
    
    date_str = str(date_str).strip()
    
    # Try to extract year if it's a complex date
    year_match = re.search(r'\b(1[4-9]\d{2}|20\d{2})\b', date_str)
    if year_match:
        year = year_match.group(1)
        date_str = f"YEAR {year}"
    
    # Normalize the date string
    date_str = re.sub(r'[^A-Za-z0-9\s.,-]', '', date_str)
    date_str = date_str.upper().strip()
    
    # Truncate or pad to 18 characters
    if len(date_str) > 18:
        date_str = date_str[:18]
    else:
        date_str = date_str.ljust(18)
    
    return date_str

def get_patterns():
    """
    Return the pattern dictionary used for character encoding.
    """
    return {
        'A': [[1,0,1], [0,1,0], [1,0,1]],
        'B': [[1,1,0], [1,0,1], [0,1,1]],
        'C': [[1,0,0], [0,1,0], [0,0,1]],
        'D': [[0,1,1], [1,0,0], [1,1,0]],
        'E': [[1,1,1], [0,1,0], [1,1,1]],
        'F': [[1,1,1], [0,1,0], [1,0,0]],
        'G': [[1,1,1], [1,0,0], [1,1,1]],
        'H': [[1,0,1], [1,1,1], [1,0,1]],
        'I': [[1,1,1], [0,1,0], [1,1,1]],
        'J': [[0,0,1], [0,0,1], [1,1,1]],
        'K': [[1,0,1], [1,1,0], [1,0,1]],
        'L': [[1,0,0], [1,0,0], [1,1,1]],
        'M': [[1,0,1], [1,1,1], [1,0,1]],
        'N': [[1,1,0], [1,0,1], [0,1,1]],
        'O': [[1,1,1], [1,0,1], [1,1,1]],
        'P': [[1,1,1], [1,1,0], [1,0,0]],
        'Q': [[1,1,1], [1,0,1], [1,1,0]],
        'R': [[1,1,0], [1,1,1], [1,0,1]],
        'S': [[0,1,1], [0,1,0], [1,1,0]],
        'T': [[1,1,1], [0,1,0], [0,1,0]],
        'U': [[1,0,1], [1,0,1], [1,1,1]],
        'V': [[1,0,1], [1,0,1], [0,1,0]],
        'W': [[1,0,1], [1,1,1], [1,0,1]],
        'X': [[1,0,1], [0,1,0], [1,0,1]],
        'Y': [[1,0,1], [0,1,0], [0,1,0]],
        'Z': [[1,1,1], [0,1,0], [1,1,1]],
        '0': [[1,1,1], [1,0,1], [1,1,1]],
        '1': [[0,1,0], [1,1,0], [0,1,0]],
        '2': [[1,1,1], [0,1,1], [1,1,1]],
        '3': [[1,1,1], [0,1,1], [1,1,1]],
        '4': [[1,0,1], [1,1,1], [0,0,1]],
        '5': [[1,1,1], [1,1,0], [1,1,1]],
        '6': [[1,1,1], [1,1,0], [1,1,1]],
        '7': [[1,1,1], [0,0,1], [0,0,1]],
        '8': [[1,1,1], [1,1,1], [1,1,1]],
        '9': [[1,1,1], [1,1,1], [0,0,1]],
        ' ': [[0,1,0], [0,0,0], [0,1,0]],  # Space
        '.': [[0,0,0], [0,0,0], [0,1,0]],  # Period
        ',': [[0,0,0], [0,0,0], [0,1,1]],  # Comma
        '-': [[0,0,0], [1,1,1], [0,0,0]],  # Hyphen
    }

def create_pattern_image(text, cell_size=20):
    """
    Create a pattern image from text using 3x3 patterns for each character.
    """
    patterns = get_patterns()
    
    # Calculate image dimensions
    num_chars = len(text)
    img_width = num_chars * 3 * cell_size
    img_height = 3 * cell_size
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw patterns
    for char_idx, char in enumerate(text):
        if char in patterns:
            pattern = patterns[char]
            
            # Draw 3x3 pattern
            for row in range(3):
                for col in range(3):
                    x1 = (char_idx * 3 + col) * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    color = 'black' if pattern[row][col] == 1 else 'white'
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')
        else:
            # Unknown character - use error pattern
            for row in range(3):
                for col in range(3):
                    x1 = (char_idx * 3 + col) * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    color = 'red' if (row + col) % 2 == 0 else 'white'
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')
    
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
    chars_per_row = 6  # 8 characters per row to fit nicely in 1000px
    cell_size = 20  # Size of each pattern cell
    pattern_width = 3 * cell_size  # 3x3 pattern
    char_width = 60  # Space for character display
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

def process_csv_to_patterns(csv_file_path, output_dir="pattern_images"):
    """
    Process CSV file and generate pattern images for each entry.
    """
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
    # Look for common column names
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
        print(f"Processing row {idx + 1}/{len(df)}")
        
        # Extract and normalize data
        lat_data = row[lat_col] if lat_col else None
        lng_data = row[lng_col] if lng_col else None
        name_data = row[name_col] if name_col else None
        date_data = row[date_col] if date_col else None
        
        # Normalize each field to 18 characters
        normalized_lat = normalize_coordinate(lat_data, 'lat')
        normalized_lng = normalize_coordinate(lng_data, 'lng')
        normalized_name = normalize_text(name_data)
        normalized_date = normalize_date(date_data)
        
        print(f"  Normalized lat: '{normalized_lat}'")
        print(f"  Normalized lng: '{normalized_lng}'")
        print(f"  Normalized name: '{normalized_name}'")
        print(f"  Normalized date: '{normalized_date}'")
        
        # Generate pattern images
        try:
            # Create images for each field
            lat_img = create_pattern_image(normalized_lat, cell_size=20)
            lng_img = create_pattern_image(normalized_lng, cell_size=20)
            name_img = create_pattern_image(normalized_name, cell_size=20)
            date_img = create_pattern_image(normalized_date, cell_size=20)
            
            # Save images with ID-based filenames
            row_id = idx + 1  # Use 1-based indexing for IDs
            lat_img.save(os.path.join(output_dir, f"{row_id:03d}-lat.png"))
            lng_img.save(os.path.join(output_dir, f"{row_id:03d}-lng.png"))
            name_img.save(os.path.join(output_dir, f"{row_id:03d}-name.png"))
            date_img.save(os.path.join(output_dir, f"{row_id:03d}-date.png"))
            
            print(f"  ✓ Generated images for entry {row_id}")
            
        except Exception as e:
            print(f"  ✗ Error generating images for row {idx}: {e}")
    
    print(f"\nProcessing complete! Images saved in '{output_dir}' directory")

if __name__ == "__main__":
    # First, create the dictionary image
    print("Creating pattern dictionary image...")
    create_dictionary_image("pattern_dictionary.png", 1000)
    
    # Then process the CSV file
    csv_file_path = "data.csv"
    
    # Process the CSV and generate pattern images
    process_csv_to_patterns(csv_file_path)
    
    print("\nPattern generation complete!")
    print("Files created:")
    print("- pattern_dictionary.png (1000x1000 reference image)")
    print("- Pattern images for each CSV entry:")
    print("  - {id}-lat.png (latitude pattern)")
    print("  - {id}-lng.png (longitude pattern)")  
    print("  - {id}-name.png (name pattern)")
    print("  - {id}-date.png (date pattern)")