import cv2
import numpy as np
import os
import json
import math

class ImprovedHexClassifier:
    def __init__(self, tiles_folder="../AllTiles"):
        self.tiles_folder = tiles_folder
        self.reference_tiles = {}
        self.load_reference_tiles()
    
    def load_reference_tiles(self):
        """Load all reference tiles from the AllTiles folder"""
        print("Loading reference tiles...")
        for filename in os.listdir(self.tiles_folder):
            if filename.endswith('.png'):
                tile_num = filename.split('.')[0]
                tile_path = os.path.join(self.tiles_folder, filename)
                tile_img = cv2.imread(tile_path)
                if tile_img is not None:
                    self.reference_tiles[tile_num] = tile_img
                    print(f"Loaded tile {tile_num}")
        print(f"Total reference tiles loaded: {len(self.reference_tiles)}")
    
    def analyze_original_map_structure(self, map_img):
        """Analyze the original map to understand its true structure"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(map_img, cv2.COLOR_BGR2HSV)
        
        # Find the actual content area (non-white/background)
        # Create a mask for non-white pixels
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([179, 255, 245])  # Exclude pure white
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find contours to get the actual map boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main map area)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(main_contour)
            
            print(f"Map content area: x={x}, y={y}, w={w}, h={h}")
            return x, y, w, h
        
        return 0, 0, map_img.shape[1], map_img.shape[0]
    
    def detect_hex_grid_from_content(self, map_img):
        """Detect hexagonal grid based on actual content analysis"""
        # Get reference tile dimensions
        ref_tile_key = next(iter(self.reference_tiles))
        ref_tile = self.reference_tiles[ref_tile_key]
        tile_height, tile_width = ref_tile.shape[:2]
        
        print(f"Reference tile dimensions: {tile_width}x{tile_height}")
        
        # Analyze the map structure
        content_x, content_y, content_w, content_h = self.analyze_original_map_structure(map_img)
        
        # Calculate proper hexagonal spacing
        h_spacing = int(tile_width * 0.75)  # 3/4 of tile width
        v_spacing = int(tile_height * 0.866)  # sqrt(3)/2 of tile height
        
        print(f"Calculated spacing: h={h_spacing}, v={v_spacing}")
        print(f"Content area: {content_x}, {content_y}, {content_w}, {content_h}")
        
        # Calculate grid dimensions based on content area
        # The map appears to be roughly diamond-shaped
        center_x = content_x + content_w // 2
        center_y = content_y + content_h // 2
        
        # Estimate the number of rows and columns in the diamond
        max_radius_x = content_w // 2
        max_radius_y = content_h // 2
        
        # Calculate how many tiles fit in each direction
        max_cols = (max_radius_x * 2) // h_spacing
        max_rows = (max_radius_y * 2) // v_spacing
        
        print(f"Estimated grid: {max_rows} rows, {max_cols} cols")
        print(f"Map center: ({center_x}, {center_y})")
        
        return {
            'tile_width': tile_width,
            'tile_height': tile_height,
            'h_spacing': h_spacing,
            'v_spacing': v_spacing,
            'content_bounds': (content_x, content_y, content_w, content_h),
            'center': (center_x, center_y),
            'max_rows': max_rows,
            'max_cols': max_cols
        }
    
    def is_position_in_diamond(self, row, col, center_row, center_col, max_radius):
        """Check if a position is within the diamond-shaped map"""
        # Calculate distance from center
        row_dist = abs(row - center_row)
        col_dist = abs(col - center_col)
        
        # Diamond shape: row_dist + col_dist <= radius
        return (row_dist + col_dist) <= max_radius
    
    def find_best_match(self, tile_region, threshold=0.8):
        """Find the best matching reference tile"""
        best_match = None
        best_score = float('inf')
        
        # Only check non-empty regions
        if np.var(tile_region) < 100:  # Low variance = likely empty
            return None, 1.0
        
        for tile_num, ref_tile in self.reference_tiles.items():
            # Resize reference tile to match extracted tile size
            if ref_tile.shape[:2] != tile_region.shape[:2]:
                ref_tile_resized = cv2.resize(ref_tile, (tile_region.shape[1], tile_region.shape[0]))
            else:
                ref_tile_resized = ref_tile
            
            # Use template matching
            result = cv2.matchTemplate(tile_region, ref_tile_resized, cv2.TM_SQDIFF_NORMED)
            min_val = np.min(result)
            
            if min_val < best_score:
                best_score = min_val
                best_match = tile_num
        
        # Only return match if confidence is good enough
        if best_score < threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def classify_hex_map_improved(self, map_image_path, output_file="improved_hex_classification.json"):
        """Improved hex map classification that matches the original structure"""
        print(f"Analyzing hexagonal map: {map_image_path}")
        
        # Load map image
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        # Analyze grid structure
        grid_info = self.detect_hex_grid_from_content(map_img)
        
        # Initialize results - create a more realistic diamond-shaped grid
        # Based on the original image, it looks like about 7 rows in a diamond pattern
        diamond_rows = 7
        center_row = diamond_rows // 2
        
        # Define the diamond pattern (tiles per row)
        diamond_pattern = []
        for row in range(diamond_rows):
            distance_from_center = abs(row - center_row)
            tiles_in_row = 7 - distance_from_center  # 4, 5, 6, 7, 6, 5, 4
            diamond_pattern.append(tiles_in_row)
        
        print(f"Diamond pattern: {diamond_pattern}")
        
        # Create the grid
        max_cols = max(diamond_pattern)
        tile_array = [[0 for _ in range(max_cols)] for _ in range(diamond_rows)]
        
        # Calculate starting position to center the diamond in the content area
        content_x, content_y, content_w, content_h = grid_info['content_bounds']
        
        # Start from the content area
        start_x = content_x
        start_y = content_y
        
        # Process each row of the diamond
        for row in range(diamond_rows):
            tiles_in_row = diamond_pattern[row]
            
            # Calculate y position
            y = start_y + row * grid_info['v_spacing']
            
            # Calculate starting x position for this row (center the tiles)
            empty_cols = max_cols - tiles_in_row
            start_col = empty_cols // 2
            
            # Calculate x offset for odd rows (hexagonal stagger)
            x_offset = (grid_info['h_spacing'] // 2) if row % 2 == 1 else 0
            
            # Process each tile in this row
            for tile_idx in range(tiles_in_row):
                col = start_col + tile_idx
                
                # Calculate x position
                x = start_x + col * grid_info['h_spacing'] + x_offset
                
                # Extract tile region
                if (x + grid_info['tile_width'] <= map_img.shape[1] and 
                    y + grid_info['tile_height'] <= map_img.shape[0] and
                    x >= 0 and y >= 0):
                    
                    tile_region = map_img[y:y+grid_info['tile_height'], x:x+grid_info['tile_width']]
                    
                    # Classify the tile
                    tile_match, confidence = self.find_best_match(tile_region)
                    
                    if tile_match:
                        tile_array[row][col] = int(tile_match)
                        print(f"Row {row}, Col {col}: Tile {tile_match} (confidence: {confidence:.4f})")
                    else:
                        print(f"Row {row}, Col {col}: No match found (confidence: {confidence:.4f})")
        
        # Save results
        results = {
            'tile_array': tile_array,
            'diamond_pattern': diamond_pattern,
            'grid_info': grid_info,
            'map_structure': 'diamond',
            'reference_tiles_used': list(self.reference_tiles.keys())
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Classification complete! Results saved to {output_file}")
        return tile_array

if __name__ == "__main__":
    # Create improved classifier
    classifier = ImprovedHexClassifier()
    
    # Classify the map with improved algorithm
    tile_array = classifier.classify_hex_map_improved("map.png")
    
    if tile_array:
        print("\nImproved Hexagonal Map Array:")
        for i, row in enumerate(tile_array):
            print(f"Row {i}: {row}")
    else:
        print("Classification failed!")
