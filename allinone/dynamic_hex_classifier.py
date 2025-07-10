import cv2
import numpy as np
import os
import json
import math

class DynamicHexClassifier:
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
    
    def create_hexagon_mask(self, width, height):
        """Create a hexagonal mask for a given width and height"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create hexagon vertices
        center_x, center_y = width // 2, height // 2
        
        # Hexagon points
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center_x + (width // 2 - 2) * math.cos(angle)
            y = center_y + (height // 2 - 2) * math.sin(angle)
            points.append([int(x), int(y)])
        
        # Fill the hexagon
        cv2.fillPoly(mask, [np.array(points)], 255)
        return mask
    
    def is_tile_present(self, tile_region, threshold=100):
        """Check if a tile is actually present at this position"""
        # Convert to grayscale
        gray = cv2.cvtColor(tile_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate the variance - empty/background regions have low variance
        variance = np.var(gray)
        
        # If variance is too low, it's likely background/empty
        return variance > threshold
    
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
    
    def detect_hex_positions(self, map_img):
        """Dynamically detect hexagonal tile positions in the image"""
        # Get reference tile dimensions
        ref_tile_key = next(iter(self.reference_tiles))
        ref_tile = self.reference_tiles[ref_tile_key]
        tile_height, tile_width = ref_tile.shape[:2]
        
        # Calculate hexagonal grid parameters
        h_spacing = int(tile_width * 0.75)  # 3/4 of tile width
        v_spacing = int(tile_height * 0.866)  # sqrt(3)/2 of tile height
        
        map_height, map_width = map_img.shape[:2]
        
        # Find all potential tile positions
        tile_positions = []
        
        # Start from top-left and scan the entire image
        row = 0
        for y in range(0, map_height - tile_height + 1, v_spacing):
            col = 0
            # Calculate x offset for hexagonal pattern (every other row is offset)
            x_offset = (h_spacing // 2) if row % 2 == 1 else 0
            
            for x in range(x_offset, map_width - tile_width + 1, h_spacing):
                # Extract tile region
                tile_region = map_img[y:y+tile_height, x:x+tile_width]
                
                # Check if a tile is present at this position
                if self.is_tile_present(tile_region):
                    tile_positions.append({
                        'row': row,
                        'col': col,
                        'x': x,
                        'y': y,
                        'region': tile_region
                    })
                
                col += 1
            row += 1
        
        return tile_positions, tile_width, tile_height, h_spacing, v_spacing
    
    def positions_to_2d_array(self, tile_positions):
        """Convert detected tile positions to a 2D array with proper indexing"""
        if not tile_positions:
            return []
        
        # Find the bounds of the grid
        max_row = max(pos['row'] for pos in tile_positions)
        max_col = max(pos['col'] for pos in tile_positions)
        
        # Create 2D array with zeros
        grid = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        # Create hexagonal mask
        if tile_positions:
            tile_width = tile_positions[0]['region'].shape[1]
            tile_height = tile_positions[0]['region'].shape[0]
            hex_mask = self.create_hexagon_mask(tile_width, tile_height)
        
        # Fill the grid with classified tiles
        for pos in tile_positions:
            row, col = pos['row'], pos['col']
            tile_region = pos['region']
            
            # Classify the tile
            tile_match, confidence = self.find_best_match(tile_region, hex_mask)
            
            if tile_match and confidence < 0.8:  # Only accept good matches
                grid[row][col] = int(tile_match)
                print(f"Row {row}, Col {col}: Tile {tile_match} (confidence: {confidence:.4f})")
            else:
                print(f"Row {row}, Col {col}: No good match (confidence: {confidence:.4f})")
        
        return grid
    
    def classify_hex_map(self, map_image_path, output_file="dynamic_hex_classification.json"):
        """Dynamically classify the hexagonal map without hardcoded dimensions"""
        print(f"Analyzing hexagonal map: {map_image_path}")
        
        # Load map image
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        # Detect tile positions dynamically
        tile_positions, tile_width, tile_height, h_spacing, v_spacing = self.detect_hex_positions(map_img)
        
        print(f"Detected {len(tile_positions)} potential tile positions")
        print(f"Tile dimensions: {tile_width}x{tile_height}")
        print(f"Grid spacing: h={h_spacing}, v={v_spacing}")
        
        # Convert to 2D array
        tile_array = self.positions_to_2d_array(tile_positions)
        
        if not tile_array:
            print("No tiles detected!")
            return None
        
        # Save results
        results = {
            'tile_array': tile_array,
            'detected_positions': len(tile_positions),
            'grid_dimensions': {
                'rows': len(tile_array),
                'cols': len(tile_array[0]) if tile_array else 0
            },
            'grid_info': {
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
        print(f"Grid size: {len(tile_array)} rows x {len(tile_array[0]) if tile_array else 0} columns")
        
        return tile_array

if __name__ == "__main__":
    # Create classifier and process the map
    classifier = DynamicHexClassifier()
    
    # Classify the hexagonal map
    tile_array = classifier.classify_hex_map("map.png")
    
    if tile_array:
        print("\nDynamic Hexagonal Map Array (0s for holes):")
        for i, row in enumerate(tile_array):
            print(f"Row {i}: {row}")
            
        # Print a visual representation showing hexagonal offset
        print("\nVisual representation (0 = hole, numbers = tile types):")
        for i, row in enumerate(tile_array):
            # Add spacing for odd rows to show hexagonal offset
            spacing = "  " if i % 2 == 1 else ""
            row_str = spacing + " ".join([f"{tile:2d}" for tile in row])
            print(f"Row {i}: {row_str}")
    else:
        print("Classification failed!")
