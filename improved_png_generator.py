import cv2
import numpy as np
import json
import os

class ImprovedPNGGenerator:
    def __init__(self, tiles_folder="AllTiles"):
        self.tiles_folder = tiles_folder
    
    def load_classification_results(self, json_file="improved_hex_classification.json"):
        """Load improved classification results"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Classification file not found: {json_file}")
            return None
    
    def create_hex_mask(self, width, height):
        """Create a hexagonal mask for a tile"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create hexagon vertices
        center_x, center_y = width // 2, height // 2
        
        # Hexagon points
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            x = center_x + (width // 2 - 2) * np.cos(angle)
            y = center_y + (height // 2 - 2) * np.sin(angle)
            points.append([int(x), int(y)])
        
        # Fill the hexagon
        cv2.fillPoly(mask, [np.array(points)], 255)
        return mask
    
    def generate_diamond_shaped_map(self, results, output_file="improved_generated_map.png"):
        """Generate a diamond-shaped map that matches the original structure"""
        if not results:
            return False
        
        tile_array = results['tile_array']
        grid_info = results['grid_info']
        diamond_pattern = results['diamond_pattern']
        
        tile_width = grid_info['tile_width']
        tile_height = grid_info['tile_height']
        h_spacing = grid_info['h_spacing']
        v_spacing = grid_info['v_spacing']
        
        # Calculate canvas size based on diamond pattern
        max_cols = max(diamond_pattern)
        num_rows = len(diamond_pattern)
        
        # Calculate canvas dimensions to fit the diamond
        canvas_width = int(max_cols * h_spacing + tile_width)
        canvas_height = int(num_rows * v_spacing + tile_height)
        
        # Create canvas with transparent background
        canvas = np.full((canvas_height, canvas_width, 4), [255, 255, 255, 0], dtype=np.uint8)
        
        # Create hexagonal mask
        hex_mask = self.create_hex_mask(tile_width, tile_height)
        
        # Place tiles according to diamond pattern
        for row in range(num_rows):
            if row >= len(tile_array):
                continue
                
            tiles_in_row = diamond_pattern[row]
            
            # Calculate y position
            y = row * v_spacing
            
            # Calculate starting x position for this row (center the tiles)
            empty_cols = max_cols - tiles_in_row
            start_col = empty_cols // 2
            
            # Calculate x offset for odd rows (hexagonal stagger)
            x_offset = (h_spacing // 2) if row % 2 == 1 else 0
            
            # Place tiles in this row
            for tile_idx in range(tiles_in_row):
                col = start_col + tile_idx
                
                if col < len(tile_array[row]):
                    tile_id = tile_array[row][col]
                    
                    if tile_id != 0:  # Skip empty tiles
                        # Calculate x position
                        x = col * h_spacing + x_offset
                        
                        # Load tile image
                        tile_path = os.path.join(self.tiles_folder, f"{tile_id}.png")
                        if os.path.exists(tile_path):
                            tile_img = cv2.imread(tile_path)
                            if tile_img is not None:
                                # Resize tile if necessary
                                if tile_img.shape[:2] != (tile_height, tile_width):
                                    tile_img = cv2.resize(tile_img, (tile_width, tile_height))
                                
                                # Apply hexagonal mask
                                tile_rgba = cv2.cvtColor(tile_img, cv2.COLOR_BGR2BGRA)
                                
                                # Apply mask to alpha channel
                                tile_rgba[:, :, 3] = hex_mask
                                
                                # Place tile on canvas
                                end_y = min(y + tile_height, canvas_height)
                                end_x = min(x + tile_width, canvas_width)
                                
                                if y < canvas_height and x < canvas_width:
                                    # Blend the tile onto the canvas
                                    tile_region = tile_rgba[0:end_y-y, 0:end_x-x]
                                    canvas_region = canvas[y:end_y, x:end_x]
                                    
                                    # Alpha blending
                                    alpha = tile_region[:, :, 3:4] / 255.0
                                    canvas[y:end_y, x:end_x] = (
                                        tile_region * alpha + 
                                        canvas_region * (1 - alpha)
                                    ).astype(np.uint8)
        
        # Convert to BGR for saving (remove alpha channel)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        
        # Save image
        cv2.imwrite(output_file, canvas_bgr)
        print(f"Improved generated map saved: {output_file}")
        return True

if __name__ == "__main__":
    # Load classification results
    generator = ImprovedPNGGenerator()
    results = generator.load_classification_results()
    
    if results:
        # Generate improved map
        if generator.generate_diamond_shaped_map(results):
            print("✅ Improved map generation complete!")
        else:
            print("❌ Failed to generate improved map")
    else:
        print("❌ No classification results found. Run improved_hex_classifier.py first.")