import cv2
import numpy as np
import os
import json
import math

class CorrectHexClassifier:
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
    
    def analyze_map_structure(self, map_img):
        """Analyze the original map to understand the correct hex structure"""
        # Get reference tile dimensions
        ref_tile_key = next(iter(self.reference_tiles))
        ref_tile = self.reference_tiles[ref_tile_key]
        tile_height, tile_width = ref_tile.shape[:2]
        
        print(f"Reference tile dimensions: {tile_width}x{tile_height}")
        
        # For pointy-top hexagons (correct orientation):
        # - Horizontal spacing: full width of hex
        # - Vertical spacing: 3/4 of height (since hexes are taller than wide)
        # - Odd rows offset by half the horizontal spacing
        
        h_spacing = tile_width  # Full width for pointy-top
        v_spacing = int(tile_height * 0.75)  # 3/4 height for vertical spacing
        
        print(f"Correct hex spacing: h={h_spacing}, v={v_spacing}")
        
        return {
            'tile_width': tile_width,
            'tile_height': tile_height,
            'h_spacing': h_spacing,
            'v_spacing': v_spacing
        }
    
    def extract_tile_precisely(self, map_img, x, y, tile_width, tile_height):
        """Extract a tile region with bounds checking"""
        map_height, map_width = map_img.shape[:2]
        
        # Ensure we don't go out of bounds
        x = max(0, min(x, map_width - tile_width))
        y = max(0, min(y, map_height - tile_height))
        
        end_x = min(x + tile_width, map_width)
        end_y = min(y + tile_height, map_height)
        
        return map_img[y:end_y, x:end_x]
    
    def find_best_tile_match(self, tile_region, confidence_threshold=0.7):
        """Find the best matching tile with improved accuracy"""
        if tile_region.size == 0:
            return None, 1.0
        
        # Check if tile region has enough content
        if np.var(tile_region) < 200:  # Increase threshold for better detection
            return None, 1.0
        
        best_match = None
        best_score = float('inf')
        
        # Try multiple matching methods for better accuracy
        for tile_num, ref_tile in self.reference_tiles.items():
            # Resize reference tile to match extracted tile size
            if ref_tile.shape[:2] != tile_region.shape[:2]:
                ref_tile_resized = cv2.resize(ref_tile, (tile_region.shape[1], tile_region.shape[0]))
            else:
                ref_tile_resized = ref_tile
            
            # Method 1: Template matching
            result = cv2.matchTemplate(tile_region, ref_tile_resized, cv2.TM_SQDIFF_NORMED)
            template_score = np.min(result)
            
            # Method 2: Color histogram comparison
            hist1 = cv2.calcHist([tile_region], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([ref_tile_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist_score = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Combined score (weighted average)
            combined_score = 0.7 * template_score + 0.3 * hist_score
            
            if combined_score < best_score:
                best_score = combined_score
                best_match = tile_num
        
        # Only return match if confidence is good enough
        if best_score < confidence_threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def scan_for_tiles(self, map_img, grid_info):
        """Scan the entire map for tiles using correct hex positioning"""
        tile_width = grid_info['tile_width']
        tile_height = grid_info['tile_height']
        h_spacing = grid_info['h_spacing']
        v_spacing = grid_info['v_spacing']
        
        map_height, map_width = map_img.shape[:2]
        
        # Calculate how many rows and columns we can fit
        max_rows = (map_height - tile_height) // v_spacing + 1
        max_cols = (map_width - tile_width) // h_spacing + 1
        
        print(f"Scanning grid: {max_rows} rows x {max_cols} cols")
        
        # Initialize results
        tiles_found = []
        
        # Scan each possible position
        for row in range(max_rows):
            for col in range(max_cols):
                # Calculate position for pointy-top hexagons
                x = col * h_spacing
                y = row * v_spacing
                
                # Offset every other row by half spacing (pointy-top hex grid)
                if row % 2 == 1:
                    x += h_spacing // 2
                
                # Extract tile region
                tile_region = self.extract_tile_precisely(map_img, x, y, tile_width, tile_height)
                
                # Classify the tile
                tile_match, confidence = self.find_best_tile_match(tile_region)
                
                if tile_match:
                    tiles_found.append({
                        'row': row,
                        'col': col,
                        'tile_id': int(tile_match),
                        'confidence': confidence,
                        'position': (x, y)
                    })
                    print(f"Found tile at ({row}, {col}): {tile_match} (confidence: {confidence:.4f})")
        
        return tiles_found
    
    def create_tile_grid(self, tiles_found):
        """Convert found tiles to a proper 2D grid"""
        if not tiles_found:
            return []
        
        # Find grid bounds
        max_row = max(tile['row'] for tile in tiles_found)
        max_col = max(tile['col'] for tile in tiles_found)
        
        # Create grid
        grid = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        # Fill grid
        for tile in tiles_found:
            grid[tile['row']][tile['col']] = tile['tile_id']
        
        return grid
    
    def classify_hex_map_correct(self, map_image_path, output_file="correct_hex_classification.json"):
        """Classify hex map with correct orientation and better accuracy"""
        print(f"Analyzing map with correct hex orientation: {map_image_path}")
        
        # Load map image
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        # Analyze map structure
        grid_info = self.analyze_map_structure(map_img)
        
        # Scan for tiles
        tiles_found = self.scan_for_tiles(map_img, grid_info)
        
        # Create grid
        tile_array = self.create_tile_grid(tiles_found)
        
        # Save results
        results = {
            'tile_array': tile_array,
            'tiles_found': len(tiles_found),
            'grid_info': grid_info,
            'hex_orientation': 'pointy-top',
            'reference_tiles_used': list(self.reference_tiles.keys())
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Classification complete! Found {len(tiles_found)} tiles")
        print(f"Grid size: {len(tile_array)} rows x {len(tile_array[0]) if tile_array else 0} cols")
        
        return tile_array

if __name__ == "__main__":
    # Create correct hex classifier
    classifier = CorrectHexClassifier()
    
    # Classify with correct orientation
    tile_array = classifier.classify_hex_map_correct("map.png")
    
    if tile_array:
        print("\nCorrect Hex Map Array:")
        for i, row in enumerate(tile_array):
            print(f"Row {i}: {row}")
    else:
        print("Classification failed!")