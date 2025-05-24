from PIL import Image, ImageDraw
import numpy as np

def create_pattern_image(text, cell_size=20):
    # Define 3x3 patterns for characters (0=white, 1=black)
    # Using simple binary encoding with good visual distinction
    patterns = {
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
        ' ': [[0,0,0], [0,0,0], [0,0,0]],  # Space
        '.': [[0,0,0], [0,0,0], [0,1,0]],  # Period
        ',': [[0,0,0], [0,0,0], [0,1,1]],  # Comma
        '-': [[0,0,0], [1,1,1], [0,0,0]],  # Hyphen
    }
    
    # Convert text to uppercase
    text = text.upper()
    
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

# Example usage
if __name__ == "__main__":
    # Input your text here
    text = "Batalha de Caiboate"
    
    # Generate image
    image = create_pattern_image(text, cell_size=30)
    
    # Save image
    image.save("pattern_output.png")
    print(f"Generated pattern for: {text}")
    print("Image saved as 'pattern_output.png'")
    
    # Show image (if running in interactive environment)
    # image.show()