"""
Character recognition functions for template matching and pattern recognition
"""

import cv2
import numpy as np
from patterns import get_patterns

def is_mostly_white(img, row, col, threshold=0.95):
    """Check if image is mostly white/empty"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    white_pixels = np.sum(gray > 85)
    total_pixels = gray.shape[0] * gray.shape[1]
    white_ratio = white_pixels / total_pixels
    print(f"Row {row}, Col {col}: White ratio = {white_ratio}")
    
    return white_ratio > threshold


def extract_cell_pattern(cell_img, grid_size=5):
    """Extract 3x3 binary pattern from a cell image using 5x5 subdivision"""
    h, w = cell_img.shape
    sub_h, sub_w = h // grid_size, w // grid_size
    
    pattern = []
    for pattern_row in range(3):
        row_pattern = []
        for pattern_col in range(3):
            # Map 3x3 pattern to center of 5x5 grid
            sub_row = pattern_row + 1
            sub_col = pattern_col + 1
            
            start_y = sub_row * sub_h
            end_y = start_y + sub_h
            start_x = sub_col * sub_w
            end_x = start_x + sub_w
            
            # Extract sub-cell with padding
            sub_cell = cell_img[start_y+10:end_y-10, start_x+10:end_x-10]
            avg_gray = np.mean(sub_cell)
            
            binary_value = 1 if avg_gray < 126 else 0
            row_pattern.append(binary_value)
        
        pattern.append(row_pattern)
    
    return pattern


def compare_patterns(pattern1, pattern2):
    """Compare two 3x3 patterns and return similarity score (0-1)"""
    if len(pattern1) != 3 or len(pattern2) != 3:
        return 0
    
    matches = sum(1 for row in range(3) for col in range(3) 
                  if pattern1[row][col] == pattern2[row][col])
    
    return matches / 9


def recognize_character_from_image(cell_img, row, col, templates=None, use_patterns=True, threshold=0.7):
    """
    Recognize character from cell image using templates or patterns
    Returns character and confidence score
    """
    # Check if mostly white first
    if is_mostly_white(cell_img, row, col):
        return ' ', 1.0
    
    # Preprocess cell
    if len(cell_img.shape) == 3:
        cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell_gray = cell_img
    
    # Resize for consistency
    cell_resized = cv2.resize(cell_gray, (400, 400), interpolation=cv2.INTER_AREA)
    
    best_char = None
    best_score = 0
    
    # Try template matching first if templates provided
    if templates:
        for template_name, template_img in templates.items():
            if cell_resized.shape[0] >= template_img.shape[0] and cell_resized.shape[1] >= template_img.shape[1]:
                result = cv2.matchTemplate(cell_resized, template_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score and max_val >= threshold:
                    best_score = max_val
                    best_char = template_name
    
    # If template matching didn't work, try pattern matching
    if not best_char and use_patterns:
        pattern = extract_cell_pattern(cell_resized)
        patterns_dict = get_patterns()
        
        for char, char_pattern in patterns_dict.items():
            score = compare_patterns(pattern, char_pattern)
            if score > best_score and score >= threshold:
                best_score = score
                best_char = char
    
    return best_char if best_char else '?', best_score


def join_characters_to_message(grid_results, grid_rows=7, grid_cols=7):
    """Join recognized characters into a complete message"""
    message_chars = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Skip corner cells
            if ((row == 0 and col == 0) or 
                (row == 0 and col == grid_cols - 1) or 
                (row == grid_rows - 1 and col == 0) or 
                (row == grid_rows - 1 and col == grid_cols - 1)):
                continue
            
            if grid_results[row][col] is not None:
                char = grid_results[row][col].get('character', '?')
                message_chars.append(char)
            else:
                message_chars.append('?')
    
    return ''.join(message_chars)