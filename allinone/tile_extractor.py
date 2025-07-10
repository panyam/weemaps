import cv2
import numpy as np
import os
from PIL import Image
import math

def load_reference_tiles(tiles_folder):
    """Load all reference tiles from the AllTiles folder"""
    reference_tiles = {}
    for filename in os.listdir(tiles_folder):
        if filename.endswith('.png'):
            tile_id = filename.split('.')[0]
            tile_path = os.path.join(tiles_folder, filename)
            tile_img = cv2.imread(tile_path)
            if tile_img is not None:
                reference_tiles[tile_id] = tile_img
    return reference_tiles

def extract_hexagonal_tiles(image_path, output_folder="extracted_tiles"):
    """Extract hexagonal tiles from the map image"""
    # Load the main map image
    map_img = cv2.imread(image_path)
    if map_img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load reference tiles
    reference_tiles = load_reference_tiles("../AllTiles")
    
    # Convert to RGB for processing
    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    
    # Estimate hexagon size based on image dimensions
    # The hexagons appear to be roughly 32x32 pixels based on the reference tiles
    hex_width = 32
    hex_height = 32
    
    # Calculate hexagonal grid parameters
    # For hexagonal grids, horizontal spacing is about 3/4 of hex width
    # Vertical spacing alternates between rows
    h_spacing = int(hex_width * 0.75)
    v_spacing = int(hex_height * 0.87)  # sqrt(3)/2 ≈ 0.87
    
    # Find approximate grid bounds
    height, width = map_rgb.shape[:2]
    
    # Start extraction from estimated grid positions
    row = 0
    extracted_tiles = []
    
    for y in range(0, height - hex_height, v_spacing):
        col = 0
        # Offset every other row for hexagonal pattern
        x_offset = (h_spacing // 2) if row % 2 == 1 else 0
        
        for x in range(x_offset, width - hex_width, h_spacing):
            # Extract tile region
            tile_region = map_rgb[y:y+hex_height, x:x+hex_width]
            
            if tile_region.shape[0] == hex_height and tile_region.shape[1] == hex_width:
                # Find best matching tile type
                tile_type, player = identify_tile(tile_region, reference_tiles)
                
                if tile_type:
                    # Save extracted tile
                    filename = f"{row}_{col}_{tile_type}_{player}.png"
                    filepath = os.path.join(output_folder, filename)
                    
                    # Convert back to BGR for saving
                    tile_bgr = cv2.cvtColor(tile_region, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, tile_bgr)
                    
                    extracted_tiles.append({
                        'row': row,
                        'col': col,
                        'type': tile_type,
                        'player': player,
                        'filename': filename
                    })
                    
                    print(f"Extracted tile: {filename}")
            
            col += 1
        row += 1
    
    print(f"Total tiles extracted: {len(extracted_tiles)}")
    return extracted_tiles

def identify_tile(tile_region, reference_tiles):
    """Identify tile type and player based on comparison with reference tiles"""
    tile_bgr = cv2.cvtColor(tile_region, cv2.COLOR_RGB2BGR)
    
    best_match = None
    best_score = float('inf')
    
    # Compare with each reference tile
    for tile_id, ref_tile in reference_tiles.items():
        # Resize reference tile to match extracted tile size if needed
        if ref_tile.shape[:2] != tile_bgr.shape[:2]:
            ref_tile_resized = cv2.resize(ref_tile, (tile_bgr.shape[1], tile_bgr.shape[0]))
        else:
            ref_tile_resized = ref_tile
        
        # Calculate similarity using MSE
        mse = np.mean((tile_bgr.astype(float) - ref_tile_resized.astype(float)) ** 2)
        
        if mse < best_score:
            best_score = mse
            best_match = tile_id
    
    # Determine player based on color analysis
    player = identify_player(tile_region)
    
    return best_match, player

def identify_player(tile_region):
    """Identify player based on color analysis"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(tile_region, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for different players
    # These ranges might need adjustment based on actual player colors
    red_range = [(0, 100, 100), (10, 255, 255)]
    blue_range = [(100, 100, 100), (130, 255, 255)]
    green_range = [(40, 100, 100), (80, 255, 255)]
    
    # Check for red (player 1)
    red_mask = cv2.inRange(hsv, np.array(red_range[0]), np.array(red_range[1]))
    if np.sum(red_mask) > 100:  # Threshold for significant red presence
        return "1"
    
    # Check for blue (player 2)
    blue_mask = cv2.inRange(hsv, np.array(blue_range[0]), np.array(blue_range[1]))
    if np.sum(blue_mask) > 100:
        return "2"
    
    # Check for green (player 3)
    green_mask = cv2.inRange(hsv, np.array(green_range[0]), np.array(green_range[1]))
    if np.sum(green_mask) > 100:
        return "3"
    
    # If no specific player color detected, assume neutral
    return "x"

if __name__ == "__main__":
    # Extract tiles from the map
    extracted = extract_hexagonal_tiles("map.png")
    
    print(f"Tile extraction complete. {len(extracted)} tiles extracted.")
