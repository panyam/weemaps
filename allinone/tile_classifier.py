import cv2
import numpy as np
import os
import json
import math

class HexTileClassifier:
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
    
    def analyze_hexagonal_grid(self, map_image_path):
        """Analyze the map image to determine hexagonal grid layout"""
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        print(f"Map image dimensions: {map_img.shape}")
        
        # Get a reference tile size from the first available reference tile
        ref_tile_key = next(iter(self.reference_tiles))
        ref_tile = self.reference_tiles[ref_tile_key]
        tile_height, tile_width = ref_tile.shape[:2]
        
        print(f"Reference tile size: {tile_width}x{tile_height}")
        
        # Calculate hexagonal grid parameters
        # For hexagonal grids, horizontal spacing is about 3/4 of tile width
        # Vertical spacing is about sqrt(3)/2 of tile height
        h_spacing = int(tile_width * 0.75)
        v_spacing = int(tile_height * 0.866)  # sqrt(3)/2 ≈ 0.866
        
        # Determine grid dimensions by analyzing the map
        map_height, map_width = map_img.shape[:2]
        
        # Calculate grid starting position to center the grid
        start_x = (map_width % h_spacing) // 2
        start_y = (map_height % v_spacing) // 2
        
        print(f"Hex spacing: horizontal={h_spacing}, vertical={v_spacing}")
        print(f"Grid starting position: ({start_x}, {start_y})")
        
        return {
            'tile_width': tile_width,
            'tile_height': tile_height,
            'h_spacing': h_spacing,
            'v_spacing': v_spacing,
            'start_x': start_x,
            'start_y': start_y,
            'map_width': map_width,
            'map_height': map_height
        }
    
    def classify_map_tiles(self, map_image_path, output_file="tile_classification.json"):
        """Classify all tiles in the map and return a 2D array of tile indexes"""
        print(f"Analyzing map: {map_image_path}")
        
        # Load map image
        map_img = cv2.imread(map_image_path)
        if map_img is None:
            print(f"Could not load map image: {map_image_path}")
            return None
        
        # Analyze grid layout
        grid_info = self.analyze_hexagonal_grid(map_image_path)
        if grid_info is None:
            return None
        
        # Create hexagonal mask
        hex_mask = self.create_hexagon_mask(grid_info['tile_width'], grid_info['tile_height'])
        
        # Initialize results
        tile_array = []
        classification_details = []
        
        # Extract and classify tiles
        print("Extracting and classifying tiles...")
        
        row = 0
        for y in range(grid_info['start_y'], grid_info['map_height'] - grid_info['tile_height'], grid_info['v_spacing']):
            tile_row = []
            detail_row = []
            
            # Calculate x offset for hexagonal pattern (every other row is offset)
            x_offset = (grid_info['h_spacing'] // 2) if row % 2 == 1 else 0
            
            col = 0
            for x in range(grid_info['start_x'] + x_offset, grid_info['map_width'] - grid_info['tile_width'], grid_info['h_spacing']):
                # Extract tile region
                tile_region = map_img[y:y+grid_info['tile_height'], x:x+grid_info['tile_width']]
                
                # Only process if we have a complete tile
                if tile_region.shape[0] == grid_info['tile_height'] and tile_region.shape[1] == grid_info['tile_width']:
                    # Classify the tile
                    tile_match, confidence = self.find_best_match(tile_region, hex_mask)
                    
                    if tile_match:
                        tile_row.append(int(tile_match))
                        detail_row.append({
                            'tile_id': int(tile_match),
                            'confidence': float(confidence),
                            'position': {'row': row, 'col': col, 'x': x, 'y': y}
                        })
                        print(f"Row {row}, Col {col}: Tile {tile_match} (confidence: {confidence:.4f})")
                    else:
                        tile_row.append(0)  # Unknown tile
                        detail_row.append({
                            'tile_id': 0,
                            'confidence': 1.0,
                            'position': {'row': row, 'col': col, 'x': x, 'y': y}
                        })
                        print(f"Row {row}, Col {col}: Unknown tile")
                
                col += 1
            
            if tile_row:  # Only add non-empty rows
                tile_array.append(tile_row)
                classification_details.append(detail_row)
            
            row += 1
        
        # Save results
        results = {
            'tile_array': tile_array,
            'grid_info': grid_info,
            'classification_details': classification_details,
            'reference_tiles_used': list(self.reference_tiles.keys())
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Classification complete! Results saved to {output_file}")
        print(f"Grid size: {len(tile_array)} rows x {len(tile_array[0]) if tile_array else 0} columns")
        
        return tile_array

if __name__ == "__main__":
    # Create classifier and process the map
    classifier = HexTileClassifier()
    
    # Classify the map
    tile_array = classifier.classify_map_tiles("map.png")
    
    if tile_array:
        print("\nTile Array (2D):")
        for i, row in enumerate(tile_array):
            print(f"Row {i}: {row}")
    else:
        print("Classification failed!")
