import cv2
import numpy as np
import json
import os
import math

class CorrectHexGenerator:
    def __init__(self, tiles_folder="../AllTiles"):
        self.tiles_folder = tiles_folder
    
    def load_classification_results(self, json_file="correct_hex_classification.json"):
        """Load classification results"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Classification file not found: {json_file}")
            return None
    
    def create_pointy_top_hex_mask(self, width, height):
        """Create a pointy-top hexagon mask (correct orientation)"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create pointy-top hexagon vertices
        center_x, center_y = width // 2, height // 2
        
        # For pointy-top hexagon, the points are at different angles
        points = []
        for i in range(6):
            # Start at top point (angle = -π/2) and go clockwise
            angle = -math.pi/2 + i * math.pi/3
            x = center_x + (width // 2 - 2) * math.cos(angle)
            y = center_y + (height // 2 - 2) * math.sin(angle)
            points.append([int(x), int(y)])
        
        # Fill the hexagon
        cv2.fillPoly(mask, [np.array(points)], 255)
        return mask
    
    def generate_correct_hex_map(self, results, output_file="correct_generated_map.png"):
        """Generate map with correct pointy-top hexagon orientation"""
        if not results:
            return False
        
        tile_array = results['tile_array']
        grid_info = results['grid_info']
        
        tile_width = grid_info['tile_width']
        tile_height = grid_info['tile_height']
        h_spacing = grid_info['h_spacing']
        v_spacing = grid_info['v_spacing']
        
        num_rows = len(tile_array)
        max_cols = max(len(row) for row in tile_array) if tile_array else 0
        
        # Calculate canvas size
        canvas_width = int(max_cols * h_spacing + h_spacing)
        canvas_height = int(num_rows * v_spacing + tile_height)
        
        # Create canvas
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        
        # Create pointy-top hexagon mask
        hex_mask = self.create_pointy_top_hex_mask(tile_width, tile_height)
        
        # Place tiles with correct positioning
        for row in range(num_rows):
            for col in range(len(tile_array[row])):
                tile_id = tile_array[row][col]
                
                if tile_id != 0:  # Skip empty tiles
                    # Calculate position (pointy-top orientation)
                    x = col * h_spacing
                    y = row * v_spacing
                    
                    # Offset every other row by half spacing
                    if row % 2 == 1:
                        x += h_spacing // 2
                    
                    # Load tile image
                    tile_path = os.path.join(self.tiles_folder, f"{tile_id}.png")
                    if os.path.exists(tile_path):
                        tile_img = cv2.imread(tile_path)
                        if tile_img is not None:
                            # Resize if necessary
                            if tile_img.shape[:2] != (tile_height, tile_width):
                                tile_img = cv2.resize(tile_img, (tile_width, tile_height))
                            
                            # Apply hexagonal mask
                            masked_tile = cv2.bitwise_and(tile_img, tile_img, mask=hex_mask)
                            
                            # Place tile on canvas
                            end_y = min(y + tile_height, canvas_height)
                            end_x = min(x + tile_width, canvas_width)
                            
                            if y < canvas_height and x < canvas_width:
                                # Only place where mask is active
                                mask_3d = cv2.cvtColor(hex_mask, cv2.COLOR_GRAY2BGR)
                                mask_normalized = mask_3d.astype(float) / 255.0
                                
                                tile_region = masked_tile[0:end_y-y, 0:end_x-x]
                                canvas_region = canvas[y:end_y, x:end_x]
                                mask_region = mask_normalized[0:end_y-y, 0:end_x-x]
                                
                                # Blend tile onto canvas using mask
                                canvas[y:end_y, x:end_x] = (
                                    tile_region * mask_region + 
                                    canvas_region * (1 - mask_region)
                                ).astype(np.uint8)
        
        # Save the generated map
        cv2.imwrite(output_file, canvas)
        print(f"Correct hex map generated: {output_file}")
        return True
    
    def create_side_by_side_comparison(self, original_path="map.png", generated_path="correct_generated_map.png", output_path="comparison.png"):
        """Create side-by-side comparison of original and generated maps"""
        original = cv2.imread(original_path)
        generated = cv2.imread(generated_path)
        
        if original is None or generated is None:
            print("Could not load images for comparison")
            return False
        
        # Resize images to same height for comparison
        target_height = min(original.shape[0], generated.shape[0])
        
        original_resized = cv2.resize(original, (int(original.shape[1] * target_height / original.shape[0]), target_height))
        generated_resized = cv2.resize(generated, (int(generated.shape[1] * target_height / generated.shape[0]), target_height))
        
        # Create side-by-side comparison
        comparison = np.hstack([original_resized, generated_resized])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Generated", (original_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, comparison)
        print(f"Side-by-side comparison saved: {output_path}")
        return True

if __name__ == "__main__":
    # Load classification results
    generator = CorrectHexGenerator()
    results = generator.load_classification_results()
    
    if results:
        # Generate correct hex map
        if generator.generate_correct_hex_map(results):
            print("✅ Correct hex map generation complete!")
            
            # Create comparison
            generator.create_side_by_side_comparison()
            
        else:
            print("❌ Failed to generate correct hex map")
    else:
        print("❌ No classification results found. Run correct_hex_classifier.py first.")
