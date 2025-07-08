#!/usr/bin/env python3
"""
Hex Grid Renderer

Renders hexagonal grids from extracted map data for validation and visualization.
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json


class HexGridRenderer:
    """Renders hexagonal grids for map validation"""
    
    def __init__(self, tile_size: int = 64):
        self.tile_size = tile_size
        self.hex_radius = tile_size // 2
        self.hex_width = int(self.hex_radius * 2)
        self.hex_height = int(self.hex_radius * math.sqrt(3))
        self.tile_cache: Dict[int, np.ndarray] = {}
        
    def load_tile_references(self, tiles_dir: Path) -> Dict[int, np.ndarray]:
        """Load and cache tile reference images"""
        tiles = {}
        
        for tile_dir in tiles_dir.iterdir():
            if tile_dir.is_dir():
                try:
                    tile_id = int(tile_dir.name)
                    tile_image_path = tile_dir / "0.png"
                    
                    if tile_image_path.exists():
                        # Load and resize tile image
                        tile_img = cv2.imread(str(tile_image_path))
                        if tile_img is not None:
                            # Resize to standard size
                            resized = cv2.resize(tile_img, (self.tile_size, self.tile_size))
                            tiles[tile_id] = resized
                            self.tile_cache[tile_id] = resized
                            
                except ValueError:
                    continue
        
        return tiles
    
    def hex_to_pixel(self, hex_row: int, hex_col: int) -> Tuple[int, int]:
        """Convert hex grid coordinates to pixel coordinates"""
        # Hex grid to pixel conversion
        x = hex_col * (self.hex_width * 0.75)
        y = hex_row * self.hex_height + (hex_col % 2) * (self.hex_height * 0.5)
        
        return int(x), int(y)
    
    def get_hex_corners(self, center_x: int, center_y: int) -> List[Tuple[int, int]]:
        """Get the corner points of a hexagon at given center"""
        corners = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center_x + self.hex_radius * math.cos(angle)
            y = center_y + self.hex_radius * math.sin(angle)
            corners.append((int(x), int(y)))
        return corners
    
    def render_hex_grid(self, grid: List[List[int]], 
                       tile_references: Dict[int, np.ndarray],
                       confidence_grid: Optional[List[List[float]]] = None,
                       highlight_errors: bool = False) -> np.ndarray:
        """Render a hex grid as an image"""
        
        if not grid or not grid[0]:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        rows = len(grid)
        cols = max(len(row) for row in grid)
        
        # Calculate canvas size
        canvas_width = int(cols * self.hex_width * 0.75 + self.hex_width * 0.25)
        canvas_height = int(rows * self.hex_height + self.hex_height * 0.5)
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas.fill(255)  # White background
        
        # Render each hex
        for row_idx, row in enumerate(grid):
            for col_idx, tile_id in enumerate(row):
                if tile_id == 0:  # Skip empty tiles
                    continue
                
                # Calculate hex center
                center_x, center_y = self.hex_to_pixel(row_idx, col_idx)
                center_x += self.hex_radius
                center_y += self.hex_radius
                
                # Get confidence if available
                confidence = 1.0
                if confidence_grid and row_idx < len(confidence_grid) and col_idx < len(confidence_grid[row_idx]):
                    confidence = confidence_grid[row_idx][col_idx]
                
                # Draw hex tile
                self._draw_hex_tile(canvas, center_x, center_y, tile_id, 
                                  tile_references, confidence, highlight_errors)
        
        return canvas
    
    def _draw_hex_tile(self, canvas: np.ndarray, center_x: int, center_y: int, 
                      tile_id: int, tile_references: Dict[int, np.ndarray],
                      confidence: float, highlight_errors: bool):
        """Draw a single hex tile on the canvas"""
        
        # Get hex corners
        corners = self.get_hex_corners(center_x, center_y)
        corners_array = np.array(corners, dtype=np.int32)
        
        # Draw tile image if available
        if tile_id in tile_references:
            tile_img = tile_references[tile_id]
            
            # Create hex mask
            mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [corners_array], 255)
            
            # Calculate tile placement
            tile_h, tile_w = tile_img.shape[:2]
            start_y = max(0, center_y - tile_h // 2)
            end_y = min(canvas.shape[0], start_y + tile_h)
            start_x = max(0, center_x - tile_w // 2)
            end_x = min(canvas.shape[1], start_x + tile_w)
            
            # Adjust tile dimensions to fit
            tile_start_y = max(0, tile_h // 2 - center_y)
            tile_end_y = tile_start_y + (end_y - start_y)
            tile_start_x = max(0, tile_w // 2 - center_x)
            tile_end_x = tile_start_x + (end_x - start_x)
            
            if (tile_end_y > tile_start_y and tile_end_x > tile_start_x and
                end_y > start_y and end_x > start_x):
                
                # Apply tile with mask
                tile_region = tile_img[tile_start_y:tile_end_y, tile_start_x:tile_end_x]
                mask_region = mask[start_y:end_y, start_x:end_x]
                
                # Blend tile with canvas
                for c in range(3):
                    canvas[start_y:end_y, start_x:end_x, c] = np.where(
                        mask_region > 0,
                        tile_region[:, :, c],
                        canvas[start_y:end_y, start_x:end_x, c]
                    )
        
        else:
            # Draw colored hex if no tile image
            color = self._get_default_tile_color(tile_id)
            cv2.fillPoly(canvas, [corners_array], color)
        
        # Draw hex outline
        outline_color = (0, 0, 0)  # Black outline
        if highlight_errors and confidence < 0.5:
            outline_color = (0, 0, 255)  # Red for low confidence
        elif highlight_errors and confidence < 0.8:
            outline_color = (0, 165, 255)  # Orange for medium confidence
        
        cv2.polylines(canvas, [corners_array], True, outline_color, 2)
        
        # Draw confidence indicator
        if confidence < 1.0:
            self._draw_confidence_indicator(canvas, center_x, center_y, confidence)
    
    def _get_default_tile_color(self, tile_id: int) -> Tuple[int, int, int]:
        """Get default color for tile ID when no image is available"""
        # Simple color mapping based on tile ID
        colors = [
            (128, 128, 128),  # Gray
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 0),      # Red
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 255, 255),    # Cyan
        ]
        return colors[tile_id % len(colors)]
    
    def _draw_confidence_indicator(self, canvas: np.ndarray, center_x: int, center_y: int, confidence: float):
        """Draw confidence indicator on tile"""
        # Draw small circle with confidence color
        radius = 8
        if confidence < 0.3:
            color = (0, 0, 255)  # Red
        elif confidence < 0.7:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        cv2.circle(canvas, (center_x, center_y), radius, color, -1)
        cv2.circle(canvas, (center_x, center_y), radius, (0, 0, 0), 1)
    
    def create_side_by_side_comparison(self, original_path: Path, 
                                     grid: List[List[int]], 
                                     tile_references: Dict[int, np.ndarray],
                                     confidence_grid: Optional[List[List[float]]] = None) -> np.ndarray:
        """Create side-by-side comparison of original and rendered map"""
        
        # Load original image
        original = cv2.imread(str(original_path))
        if original is None:
            return np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Render extracted map
        rendered = self.render_hex_grid(grid, tile_references, confidence_grid, highlight_errors=True)
        
        # Resize to same height
        height = max(original.shape[0], rendered.shape[0])
        
        # Resize original
        aspect_ratio = original.shape[1] / original.shape[0]
        new_width = int(height * aspect_ratio)
        original_resized = cv2.resize(original, (new_width, height))
        
        # Resize rendered
        aspect_ratio = rendered.shape[1] / rendered.shape[0]
        new_width = int(height * aspect_ratio)
        rendered_resized = cv2.resize(rendered, (new_width, height))
        
        # Create side-by-side image
        total_width = original_resized.shape[1] + rendered_resized.shape[1] + 20
        comparison = np.zeros((height, total_width, 3), dtype=np.uint8)
        comparison.fill(255)
        
        # Place images
        comparison[:, :original_resized.shape[1]] = original_resized
        comparison[:, original_resized.shape[1]+20:] = rendered_resized
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(comparison, "Extracted", (original_resized.shape[1] + 30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add separator line
        cv2.line(comparison, (original_resized.shape[1] + 10, 0), 
                (original_resized.shape[1] + 10, height), (0, 0, 0), 2)
        
        return comparison
    
    def create_confidence_heatmap(self, grid: List[List[int]], 
                                confidence_grid: List[List[float]]) -> np.ndarray:
        """Create confidence heatmap visualization"""
        
        if not grid or not confidence_grid:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        rows = len(grid)
        cols = max(len(row) for row in grid)
        
        # Calculate canvas size
        canvas_width = int(cols * self.hex_width * 0.75 + self.hex_width * 0.25)
        canvas_height = int(rows * self.hex_height + self.hex_height * 0.5)
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas.fill(255)
        
        # Draw confidence heatmap
        for row_idx, row in enumerate(grid):
            for col_idx, tile_id in enumerate(row):
                if tile_id == 0:
                    continue
                
                # Get confidence
                confidence = 0.0
                if (row_idx < len(confidence_grid) and 
                    col_idx < len(confidence_grid[row_idx])):
                    confidence = confidence_grid[row_idx][col_idx]
                
                # Calculate hex center
                center_x, center_y = self.hex_to_pixel(row_idx, col_idx)
                center_x += self.hex_radius
                center_y += self.hex_radius
                
                # Get hex corners
                corners = self.get_hex_corners(center_x, center_y)
                corners_array = np.array(corners, dtype=np.int32)
                
                # Color based on confidence
                color = self._confidence_to_color(confidence)
                cv2.fillPoly(canvas, [corners_array], color)
                
                # Draw outline
                cv2.polylines(canvas, [corners_array], True, (0, 0, 0), 1)
                
                # Draw confidence text
                text = f"{confidence:.2f}"
                cv2.putText(canvas, text, (center_x - 20, center_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return canvas
    
    def _confidence_to_color(self, confidence: float) -> Tuple[int, int, int]:
        """Convert confidence value to color (BGR)"""
        # Red (low) to Green (high) gradient
        if confidence < 0.5:
            # Red to yellow
            ratio = confidence * 2
            return (0, int(255 * ratio), 255)
        else:
            # Yellow to green
            ratio = (confidence - 0.5) * 2
            return (0, 255, int(255 * (1 - ratio)))
    
    def save_validation_images(self, output_dir: Path, map_id: int,
                             original_path: Path, grid: List[List[int]],
                             tile_references: Dict[int, np.ndarray],
                             confidence_grid: Optional[List[List[float]]] = None):
        """Save all validation images for a map"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rendered map
        rendered = self.render_hex_grid(grid, tile_references, confidence_grid)
        cv2.imwrite(str(output_dir / f"map_{map_id}_rendered.png"), rendered)
        
        # Save side-by-side comparison
        comparison = self.create_side_by_side_comparison(original_path, grid, tile_references, confidence_grid)
        cv2.imwrite(str(output_dir / f"map_{map_id}_comparison.png"), comparison)
        
        # Save confidence heatmap if available
        if confidence_grid:
            heatmap = self.create_confidence_heatmap(grid, confidence_grid)
            cv2.imwrite(str(output_dir / f"map_{map_id}_confidence.png"), heatmap)


def main():
    """Test the hex grid renderer"""
    renderer = HexGridRenderer()
    
    # Test with a simple grid
    test_grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    # Mock tile references
    tile_references = {}
    for i in range(1, 10):
        # Create simple colored tiles
        tile_img = np.zeros((64, 64, 3), dtype=np.uint8)
        color = renderer._get_default_tile_color(i)
        tile_img[:, :] = color
        tile_references[i] = tile_img
    
    # Render test grid
    rendered = renderer.render_hex_grid(test_grid, tile_references)
    
    # Save test image
    cv2.imwrite("test_hex_grid.png", rendered)
    print("Test hex grid saved as test_hex_grid.png")


if __name__ == "__main__":
    main()