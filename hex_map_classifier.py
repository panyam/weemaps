import cv2
import numpy as np
import os
import json
import math

class HexMapClassifier:
    def __init__(self, tiles_folder="AllTiles"):
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
    
    def create_hexagon_mask(self, width, height):
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
    
    def is_tile_present(self, tile_region, threshold=0.3):
        """Check if a tile is actually present at this position"""
        # Convert to grayscale
        gray = cv2.cvtColor(tile_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate the variance - empty/background regions have low variance
        variance = np.var(gray)
        
        # Also check if the region is mostly one color (like background)
        mean_color = np.mean(tile_region, axis=(0, 1))
        color_variance = np.var(tile_region, axis=(0, 1))
        
        # If variance is too low, it's likely background/empty
        if variance < 100 or np.max(color_variance) < 50:
            return False
        
        return True
    
    def find_best_match(self, tile_region, hex_mask):
        """Find the best matching reference tile for a given tile region"""
        best_match = None
        best_score = float('inf')
        
        for tile_num, ref_tile in self.reference_tiles.items():
            # Resize reference tile to match the extracted tile size
            if ref_tile.shape[:2] != tile_region.shape[:2]:
                ref_tile_resized = cv2.resize(ref_tile, (tile_region.shape[1], tile_region.shape[0]))
            else:
                ref_tile_resized = ref_tile
            
            # Apply hexagonal mask to both tiles
            masked_tile = cv2.bitwise_and(tile_region, tile_region, mask=hex_mask)
            masked_ref = cv2.bitwise_and(ref_tile_resized, ref_tile_resized, mask=hex_mask)
            
            # Calculate similarity using template matching
            result = cv2.matchTemplate(masked_tile, masked_ref, cv2.TM_SQDIFF_NORMED)
            min_val = np.min(result)
            
            if min_val < best_score:
                best_score = min_val
                best_match = tile_num
        
        return best_match, best_score
    
    def get_hex_grid_layout(self):
        """Define the hexagonal grid layout - number of tiles per row"""
        # Based on the hexagonal map structure
        # This defines how many tiles are in each row for a 7x7 hex grid
        return {
            0: {'count': 3, 'start_col': 2},  # Row 0: 3 tiles, starting at col 2
            1: {'count': 4, 'start_col': 1},  # Row 1: 4 tiles, starting at col 1  
            2: {'count': 5, 'start_col': 1},  # Row 2: 5 tiles, starting at col 1
            3: {'count': 6, 'start_col': 0},  # Row 3: 6 tiles, starting at col 0 (widest)
            4: {'count': 5, 'start_col': 1},  # Row 4: 5 tiles, starting at col 1
            5: {'count': 4, 'start_col': 1},  # Row 5: 4 tiles, starting at col 1
            6: {'count': 3, 'start_col': 2},  # Row 6: 3 tiles, starting at col 2
        }
    
    def classify_hex_map(self, map_image_path, output_file="hex_map_classification.json"):
        """Classify the hexagonal map and return a 7x7 2D array with 0s for holes"""
        print(f"Analyzing hexagonal map: {map_image_path}")
        
        # Load map image
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        # Get reference tile dimensions
        ref_tile_key = next(iter(self.reference_tiles))
        ref_tile = self.reference_tiles[ref_tile_key]
        tile_height, tile_width = ref_tile.shape[:2]
        
        print(f"Reference tile size: {tile_width}x{tile_height}")
        
        # Calculate hexagonal grid parameters
        h_spacing = int(tile_width * 0.75)  # 3/4 of tile width
        v_spacing = int(tile_height * 0.866)  # sqrt(3)/2 of tile height
        
        # Create hexagonal mask
        hex_mask = self.create_hexagon_mask(tile_width, tile_height)
        
        # Get hex grid layout
        hex_layout = self.get_hex_grid_layout()
        
        # Initialize 7x7 grid with zeros
        tile_array = [[0 for _ in range(7)] for _ in range(7)]
        
        # Map dimensions
        map_height, map_width = map_img.shape[:2]
        
        # Calculate starting positions to center the hexagonal grid
        start_x = (map_width - (5 * h_spacing + tile_width)) // 2
        start_y = (map_height - (6 * v_spacing + tile_height)) // 2
        
        print(f"Grid spacing: h={h_spacing}, v={v_spacing}")
        print(f"Starting position: ({start_x}, {start_y})")
        
        # Process each row
        for row in range(7):
            row_info = hex_layout[row]
            tiles_in_row = row_info['count']
            start_col = row_info['start_col']
            
            # Calculate y position for this row
            y = start_y + row * v_spacing
            
            # Calculate x offset for odd rows (hexagonal stagger)
            x_offset = (h_spacing // 2) if row % 2 == 1 else 0
            
            # Process each tile in this row
            for tile_idx in range(tiles_in_row):
                col = start_col + tile_idx
                
                # Calculate x position
                x = start_x + col * h_spacing + x_offset
                
                # Check bounds
                if (x + tile_width <= map_width and 
                    y + tile_height <= map_height and 
                    x >= 0 and y >= 0):
                    
                    # Extract tile region
                    tile_region = map_img[y:y+tile_height, x:x+tile_width]
                    
                    # Check if tile is present
                    if self.is_tile_present(tile_region):
                        # Classify the tile
                        tile_match, confidence = self.find_best_match(tile_region, hex_mask)
                        
                        if tile_match and confidence < 0.8:  # Only accept good matches
                            tile_array[row][col] = int(tile_match)
                            print(f"Row {row}, Col {col}: Tile {tile_match} (confidence: {confidence:.4f})")
                        else:
                            print(f"Row {row}, Col {col}: No good match found")
                    else:
                        print(f"Row {row}, Col {col}: No tile present")
        
        # Save results
        results = {
            'tile_array': tile_array,
            'hex_layout': hex_layout,
            'grid_info': {
                'rows': 7,
                'cols': 7,
                'tile_width': tile_width,
                'tile_height': tile_height,
                'h_spacing': h_spacing,
                'v_spacing': v_spacing
            },
            'reference_tiles_used': list(self.reference_tiles.keys())
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Classification complete! Results saved to {output_file}")
        return tile_array

if __name__ == "__main__":
    # Create classifier and process the map
    classifier = HexMapClassifier()
    
    # Classify the hexagonal map
    tile_array = classifier.classify_hex_map("map.png")
    
    if tile_array:
        print("\nHexagonal Map Array (7x7 with 0s for holes):")
        for i, row in enumerate(tile_array):
            print(f"Row {i}: {row}")
            
        # Print a visual representation
        print("\nVisual representation (0 = hole, numbers = tile types):")
        for i, row in enumerate(tile_array):
            # Add spacing for odd rows to show hexagonal offset
            spacing = "  " if i % 2 == 1 else ""
            row_str = spacing + " ".join([f"{tile:2d}" for tile in row])
            print(f"Row {i}: {row_str}")
    else:
        print("Classification failed!")