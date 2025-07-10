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

def create_hexagon_mask(width, height):
    """Create a hexagonal mask for a given width and height"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create hexagon vertices
    center_x, center_y = width // 2, height // 2
    
    # Hexagon points (roughly)
    points = []
    for i in range(6):
        angle = i * math.pi / 3
        x = center_x + (width // 2 - 2) * math.cos(angle)
        y = center_y + (height // 2 - 2) * math.sin(angle)
        points.append([int(x), int(y)])
    
    # Fill the hexagon
    cv2.fillPoly(mask, [np.array(points)], 255)
    return mask

def detect_hexagonal_grid(map_img):
    """Detect the hexagonal grid pattern in the map image"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find hexagonal shapes
    hex_contours = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly hexagonal (4-8 sides due to approximation)
        if len(approx) >= 4 and len(approx) <= 8:
            area = cv2.contourArea(contour)
            if area > 200:  # Filter out small shapes
                hex_contours.append(contour)
    
    return hex_contours

def extract_hexagonal_tiles_improved(image_path, output_folder="extracted_tiles"):
    """Extract hexagonal tiles from the map image with improved boundary detection"""
    # Load the main map image
    map_img = cv2.imread(image_path)
    if map_img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load reference tiles to get tile dimensions
    reference_tiles = load_reference_tiles("../AllTiles")
    if not reference_tiles:
        print("No reference tiles found!")
        return
    
    # Get reference tile dimensions
    ref_tile_key = next(iter(reference_tiles))
    ref_tile = reference_tiles[ref_tile_key]
    tile_height, tile_width = ref_tile.shape[:2]
    
    print(f"Reference tile dimensions: {tile_width}x{tile_height}")
    
    # Calculate hexagonal grid parameters
    # For hexagonal tiles, the horizontal distance between centers is 3/4 of width
    # The vertical distance between rows is sqrt(3)/2 of height
    hex_width = tile_width
    hex_height = tile_height
    
    # Spacing between tile centers
    h_spacing = int(hex_width * 0.75)  # 3/4 of tile width
    v_spacing = int(hex_height * 0.866)  # sqrt(3)/2 of tile height
    
    print(f"Hexagonal spacing: h={h_spacing}, v={v_spacing}")
    
    # Create hexagonal mask
    hex_mask = create_hexagon_mask(hex_width, hex_height)
    
    # Find grid bounds
    height, width = map_img.shape[:2]
    
    # Calculate grid starting position to center the grid
    start_x = (width % h_spacing) // 2
    start_y = (height % v_spacing) // 2
    
    extracted_tiles = []
    row = 0
    
    # Extract tiles using hexagonal grid pattern
    for y in range(start_y, height - hex_height, v_spacing):
        col = 0
        # Offset every other row for hexagonal pattern
        x_offset = (h_spacing // 2) if row % 2 == 1 else 0
        
        for x in range(start_x + x_offset, width - hex_width, h_spacing):
            # Extract tile region
            tile_region = map_img[y:y+hex_height, x:x+hex_width]
            
            # Check if we have a complete tile
            if tile_region.shape[0] == hex_height and tile_region.shape[1] == hex_width:
                # Apply hexagonal mask to extract only the hexagonal part
                masked_tile = cv2.bitwise_and(tile_region, tile_region, mask=hex_mask)
                
                # Find best matching tile type
                tile_type, player = identify_tile_improved(masked_tile, reference_tiles, hex_mask)
                
                if tile_type:
                    # Save extracted tile
                    filename = f"{row}_{col}_{tile_type}_{player}.png"
                    filepath = os.path.join(output_folder, filename)
                    
                    # Save the masked tile (hexagonal shape)
                    cv2.imwrite(filepath, masked_tile)
                    
                    extracted_tiles.append({
                        'row': row,
                        'col': col,
                        'type': tile_type,
                        'player': player,
                        'filename': filename,
                        'position': {'x': x, 'y': y}
                    })
                    
                    print(f"Extracted tile: {filename} at ({x}, {y})")
            
            col += 1
        row += 1
    
    print(f"Total tiles extracted: {len(extracted_tiles)}")
    return extracted_tiles

def identify_tile_improved(tile_region, reference_tiles, hex_mask):
    """Identify tile type and player with improved hexagonal matching"""
    best_match = None
    best_score = float('inf')
    
    # Compare with each reference tile
    for tile_id, ref_tile in reference_tiles.items():
        # Resize reference tile to match extracted tile size if needed
        if ref_tile.shape[:2] != tile_region.shape[:2]:
            ref_tile_resized = cv2.resize(ref_tile, (tile_region.shape[1], tile_region.shape[0]))
        else:
            ref_tile_resized = ref_tile
        
        # Apply the same hexagonal mask to the reference tile
        masked_ref = cv2.bitwise_and(ref_tile_resized, ref_tile_resized, mask=hex_mask)
        
        # Calculate similarity using template matching (more robust than MSE)
        result = cv2.matchTemplate(tile_region, masked_ref, cv2.TM_SQDIFF_NORMED)
        min_val = np.min(result)
        
        if min_val < best_score:
            best_score = min_val
            best_match = tile_id
    
    # Determine player based on color analysis
    player = identify_player_improved(tile_region, hex_mask)
    
    return best_match, player

def identify_player_improved(tile_region, hex_mask):
    """Identify player based on color analysis with hexagonal masking"""
    # Apply hexagonal mask to focus on the actual tile content
    masked_tile = cv2.bitwise_and(tile_region, tile_region, mask=hex_mask)
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(masked_tile, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for different players (adjusted for game colors)
    # Red player
    red_range1 = [(0, 100, 100), (10, 255, 255)]
    red_range2 = [(170, 100, 100), (180, 255, 255)]
    
    # Blue player
    blue_range = [(100, 100, 100), (130, 255, 255)]
    
    # Green player (if any)
    green_range = [(40, 100, 100), (80, 255, 255)]
    
    # Check for red (player 1)
    red_mask1 = cv2.inRange(hsv, np.array(red_range1[0]), np.array(red_range1[1]))
    red_mask2 = cv2.inRange(hsv, np.array(red_range2[0]), np.array(red_range2[1]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    if np.sum(red_mask) > 50:  # Threshold for significant red presence
        return "1"
    
    # Check for blue (player 2)
    blue_mask = cv2.inRange(hsv, np.array(blue_range[0]), np.array(blue_range[1]))
    if np.sum(blue_mask) > 50:
        return "2"
    
    # Check for green (player 3)
    green_mask = cv2.inRange(hsv, np.array(green_range[0]), np.array(green_range[1]))
    if np.sum(green_mask) > 50:
        return "3"
    
    # If no specific player color detected, assume neutral
    return "x"

if __name__ == "__main__":
    # Extract tiles from the map
    extracted = extract_hexagonal_tiles_improved("map.png")
    
    if extracted:
        print(f"Tile extraction complete. {len(extracted)} tiles extracted.")
        
        # Create a summary of the extracted tiles
        tile_grid = {}
        for tile in extracted:
            row, col = tile['row'], tile['col']
            if row not in tile_grid:
                tile_grid[row] = {}
            tile_grid[row][col] = tile['type']
        
        # Print grid summary
        print("\nTile Grid Summary:")
        for row in sorted(tile_grid.keys()):
            row_tiles = []
            for col in sorted(tile_grid[row].keys()):
                row_tiles.append(tile_grid[row][col])
            print(f"Row {row}: {row_tiles}")
    else:
        print("No tiles extracted!")
