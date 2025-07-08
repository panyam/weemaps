#!/usr/bin/env python3
"""
WeeWar Map Extractor

Reverse engineers WeeWar maps from preview images by detecting hexagonal grid structure
and classifying tiles through visual analysis.
"""

from ipdb import set_trace
import cv2
import numpy as np
import json
import os
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math
from hex_grid_renderer import HexGridRenderer


def debug():
    set_trace(context=21)

@dataclass
class TileInfo:
    """Information about a tile type"""
    id: int
    name: str
    image_path: str
    reference_image: Optional[np.ndarray] = None
    dominant_color: Optional[Tuple[int, int, int]] = None


@dataclass
class HexCell:
    """Represents a hexagonal cell in the grid"""
    row: int
    col: int
    center_x: float
    center_y: float
    tile_id: int = 0
    confidence: float = 0.0


class MapExtractor:
    """Main class for extracting data from preview images"""
    def __init__(self, data_dir: str = "../"):
        self.data_dir = Path(data_dir)
        self.tile_references: Dict[int, TileInfo] = {}
        self.renderer = HexGridRenderer()
    
    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Calculate dominant color of an image"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use k-means clustering to find dominant color
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # Get the most frequent cluster center
        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant_cluster = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_cluster]
        
        return tuple(map(int, dominant_color))
    
    def detect_hex_grid(self, image: np.ndarray, debug_mode: bool = False) -> List[HexCell]:
        """Detect hexagonal grid structure in the image"""
        if debug_mode:
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "01_original.png"), image)
        
        # Try multiple detection methods
        hex_centers = []
        
        # Method 1: Enhanced edge detection
        hex_centers = self._detect_hex_edges(image, debug_mode)
        
        # Method 2: Template matching if edge detection fails
        if len(hex_centers) < 10:
            print("Edge detection failed, trying template matching...")
            hex_centers = self._detect_hex_template_improved(image, debug_mode)
        
        # Method 3: Grid-based detection if others fail
        if len(hex_centers) < 10:
            print("Template matching failed, trying grid-based detection...")
            hex_centers = self._detect_hex_grid_pattern(image, debug_mode)
        
        if debug_mode:
            self._save_debug_centers(image, hex_centers, debug_dir / "final_centers.png")
        
        # Convert centers to grid structure
        return self._organize_hex_centers(hex_centers)
    
    def _detect_hex_edges(self, image: np.ndarray, debug_mode: bool = False) -> List[Tuple[int, int]]:
        """Enhanced edge detection for hexagons"""
        debug_dir = Path("debug_images") if debug_mode else None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if debug_mode:
            cv2.imwrite(str(debug_dir / "02_grayscale.png"), gray)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        if debug_mode:
            cv2.imwrite(str(debug_dir / "03_enhanced.png"), enhanced)
        
        # Use the best edge detection
        edges = cv2.Canny(enhanced, 30, 90)
        if debug_mode:
            cv2.imwrite(str(debug_dir / "04_edges_best.png"), edges)
        
        # Try different contour detection methods
        hex_centers = []
        
        # Method 1: External contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hex_centers.extend(self._extract_hex_centers_from_contours(contours, debug_mode, debug_dir, "external"))
        
        # Method 2: All contours if we don't have enough
        if len(hex_centers) < 20:
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            hex_centers.extend(self._extract_hex_centers_from_contours(contours, debug_mode, debug_dir, "list"))
        
        # Method 3: Try different approximation
        if len(hex_centers) < 20:
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hex_centers.extend(self._extract_hex_centers_from_contours(contours, debug_mode, debug_dir, "tree"))
        
        # Remove duplicates
        hex_centers = self._remove_duplicate_centers(hex_centers)
        
        return hex_centers
    
    def _extract_hex_centers_from_contours(self, contours, debug_mode, debug_dir, method_name):
        """Extract hex centers from contours with different criteria"""
        hex_centers = []
        
        for contour in contours:
            # Filter by area with more flexible range
            area = cv2.contourArea(contour)
            if area < 50 or area > 20000:  # More flexible area range
                continue
            
            # Try multiple approximation levels
            for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if roughly hexagonal (more flexible)
                if 4 <= len(approx) <= 10:  # More flexible polygon sides
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Additional validation: check if it's roughly circular/hexagonal
                        if self._is_roughly_hexagonal(contour):
                            hex_centers.append((cx, cy))
                            break  # Found a good approximation, no need to try others
        
        if debug_mode and hex_centers:
            debug_img = cv2.imread(str(debug_dir / "01_original.png"))
            for x, y in hex_centers:
                cv2.circle(debug_img, (x, y), 3, (255, 0, 0), -1)
            cv2.imwrite(str(debug_dir / f"centers_{method_name}.png"), debug_img)
        
        return hex_centers
    
    def _is_roughly_hexagonal(self, contour):
        """Check if contour is roughly hexagonal based on compactness"""
        area = cv2.contourArea(contour)
        if area == 0:
            return False
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        
        # Compactness ratio (4π*area/perimeter²)
        compactness = (4 * math.pi * area) / (perimeter * perimeter)
        
        # Hexagons have compactness around 0.9, circles have 1.0
        # Allow flexible range for detection
        return 0.3 < compactness < 1.0
    
    def _remove_duplicate_centers(self, centers, min_distance=20):
        """Remove duplicate hex centers that are too close"""
        if not centers:
            return []
        
        unique_centers = []
        for center in centers:
            is_duplicate = False
            for existing in unique_centers:
                distance = math.sqrt((center[0] - existing[0])**2 + (center[1] - existing[1])**2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_centers.append(center)
        
        return unique_centers
    
    def _detect_hex_template_improved(self, image: np.ndarray, debug_mode: bool = False) -> List[Tuple[int, int]]:
        """Improved template matching for hex detection"""
        debug_dir = Path("debug_images") if debug_mode else None
        
        # Get reference tiles for templates
        if not self.tile_references:
            return []
        
        # Use multiple reference tiles as templates
        all_matches = []
        
        for tile_id, tile_info in self.tile_references.items():
            if tile_info.reference_image is None:
                continue
            
            template = tile_info.reference_image
            
            # Try multiple scales
            for scale in [0.8, 1.0, 1.2, 1.5, 2.0]:
                h, w = template.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                
                if new_h > image.shape[0] or new_w > image.shape[1]:
                    continue
                
                scaled_template = cv2.resize(template, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find good matches
                threshold = 0.4  # Lower threshold for better detection
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    center_x = x + new_w // 2
                    center_y = y + new_h // 2
                    confidence = result[y, x]
                    all_matches.append((center_x, center_y, confidence))
        
        # Remove duplicates
        unique_matches = self._non_max_suppression(all_matches, min_distance=30)
        
        if debug_mode and unique_matches:
            self._save_debug_centers(image, [(x, y) for x, y, _ in unique_matches], 
                                   debug_dir / "template_matches.png")
        
        return [(x, y) for x, y, _ in unique_matches]
    
    def _detect_hex_grid_pattern(self, image: np.ndarray, debug_mode: bool = False) -> List[Tuple[int, int]]:
        """Pattern-based hex grid detection using expected grid structure"""
        debug_dir = Path("debug_images") if debug_mode else None
        
        # Estimate hex size from image dimensions and expected tile count
        # This is a fallback method using known information
        height, width = image.shape[:2]
        
        # Rough estimate: assume hexes are roughly square in bounding box
        estimated_hex_size = min(width, height) // 8  # Rough guess
        
        # Generate candidate hex centers in a hex grid pattern
        hex_centers = []
        
        # Hex grid geometry
        hex_width = estimated_hex_size * 1.5
        hex_height = estimated_hex_size * math.sqrt(3)
        
        rows = int(height / hex_height) + 1
        cols = int(width / hex_width) + 1
        
        for row in range(rows):
            for col in range(cols):
                # Calculate hex center
                x = col * hex_width + (row % 2) * (hex_width / 2)
                y = row * hex_height
                
                # Check if position is within image bounds
                if 0 <= x < width and 0 <= y < height:
                    # Validate this position by checking if it looks like a hex tile
                    if self._validate_hex_position(image, int(x), int(y), estimated_hex_size):
                        hex_centers.append((int(x), int(y)))
        
        if debug_mode:
            self._save_debug_centers(image, hex_centers, debug_dir / "grid_pattern.png")
        
        return hex_centers
    
    def _validate_hex_position(self, image: np.ndarray, x: int, y: int, hex_size: int) -> bool:
        """Validate if a position contains a hex tile"""
        # Extract region around position
        half_size = hex_size // 2
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(image.shape[1], x + half_size), min(image.shape[0], y + half_size)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        region = image[y1:y2, x1:x2]
        
        # Check if region has reasonable color variation (not empty/background)
        if region.size == 0:
            return False
        
        # Calculate color statistics
        mean_color = np.mean(region, axis=(0, 1))
        std_color = np.std(region, axis=(0, 1))
        
        # Valid hex should have some color variation but not too much noise
        return np.sum(std_color) > 10 and np.sum(std_color) < 200
    
    def _save_debug_centers(self, image: np.ndarray, centers: List[Tuple[int, int]], path: Path):
        """Save debug image with detected centers marked"""
        debug_img = image.copy()
        for x, y in centers:
            cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(debug_img, (x, y), 10, (0, 0, 255), 2)
        cv2.imwrite(str(path), debug_img)
    
    def _detect_hex_grid_template(self, image: np.ndarray) -> List[HexCell]:
        """Alternative hex detection using template matching"""
        # Use one of the reference tiles as a template
        if not self.tile_references:
            return []
        
        # Get a reference tile image
        reference_tile = next(iter(self.tile_references.values()))
        if reference_tile.reference_image is None:
            return []
        
        template = reference_tile.reference_image
        
        # Multi-scale template matching
        scales = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
        all_matches = []
        
        for scale in scales:
            # Resize template
            h, w = template.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_template = cv2.resize(template, (new_w, new_h))
            
            # Template matching
            result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find peaks
            threshold = 0.3
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                # Adjust for template center
                center_x = x + new_w // 2
                center_y = y + new_h // 2
                confidence = result[y, x]
                all_matches.append((center_x, center_y, confidence))
        
        # Remove duplicate matches (non-maximum suppression)
        unique_matches = self._non_max_suppression(all_matches)
        
        # Convert to hex centers
        hex_centers = [(x, y) for x, y, _ in unique_matches]
        return self._organize_hex_centers(hex_centers)
    
    def _non_max_suppression(self, matches: List[Tuple[int, int, float]], 
                           min_distance: int = 40) -> List[Tuple[int, int, float]]:
        """Remove duplicate matches that are too close together"""
        if not matches:
            return []
        
        # Sort by confidence (descending)
        matches = sorted(matches, key=lambda x: x[2], reverse=True)
        
        filtered = []
        for match in matches:
            x, y, conf = match
            
            # Check if this match is too close to any already accepted match
            too_close = False
            for fx, fy, _ in filtered:
                distance = math.sqrt((x - fx)**2 + (y - fy)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(match)
        
        return filtered
    
    def _organize_hex_centers(self, centers: List[Tuple[int, int]]) -> List[HexCell]:
        """Organize hex centers into a grid structure"""
        if not centers:
            return []
        
        # Sort centers by y-coordinate first, then x-coordinate
        centers = sorted(centers, key=lambda p: (p[1], p[0]))
        
        # Group into rows based on y-coordinate
        rows = []
        current_row = [centers[0]]
        row_threshold = 20  # Pixels tolerance for same row
        
        for i in range(1, len(centers)):
            if abs(centers[i][1] - current_row[0][1]) <= row_threshold:
                current_row.append(centers[i])
            else:
                rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [centers[i]]
        
        if current_row:
            rows.append(sorted(current_row, key=lambda p: p[0]))
        
        # Convert to HexCell objects
        hex_cells = []
        for row_idx, row in enumerate(rows):
            for col_idx, (x, y) in enumerate(row):
                hex_cells.append(HexCell(
                    row=row_idx,
                    col=col_idx,
                    center_x=x,
                    center_y=y
                ))
        
        return hex_cells
    
    def classify_tiles(self, image: np.ndarray, hex_cells: List[HexCell]) -> List[HexCell]:
        """Classify each hex cell to determine tile type"""
        # Estimate hex radius from grid spacing
        if len(hex_cells) < 2:
            return hex_cells
        
        # Calculate average distance between adjacent cells
        distances = []
        for i in range(len(hex_cells) - 1):
            for j in range(i + 1, len(hex_cells)):
                cell1, cell2 = hex_cells[i], hex_cells[j]
                dist = math.sqrt((cell1.center_x - cell2.center_x)**2 + 
                               (cell1.center_y - cell2.center_y)**2)
                if dist > 0:
                    distances.append(dist)
        
        if not distances:
            return hex_cells
        
        # Use median distance as estimate for hex spacing
        hex_spacing = np.median(distances)
        hex_radius = int(hex_spacing * 0.4)  # Approximate radius
        
        # Extract and classify each hex region
        for cell in hex_cells:
            # Extract hex region
            x1 = max(0, int(cell.center_x - hex_radius))
            y1 = max(0, int(cell.center_y - hex_radius))
            x2 = min(image.shape[1], int(cell.center_x + hex_radius))
            y2 = min(image.shape[0], int(cell.center_y + hex_radius))
            
            hex_region = image[y1:y2, x1:x2]
            
            if hex_region.size > 0:
                # Classify using template matching
                best_match_id, confidence = self._classify_hex_region(hex_region)
                cell.tile_id = best_match_id
                cell.confidence = confidence
        
        return hex_cells
    
    def _classify_hex_region(self, region: np.ndarray) -> Tuple[int, float]:
        """Classify a hex region using template matching"""
        best_match_id = 0
        best_confidence = 0.0
        
        # Try template matching with each reference tile
        for tile_id, tile_info in self.tile_references.items():
            if tile_info.reference_image is None:
                continue
            
            template = tile_info.reference_image
            
            # Resize template to match region size
            h, w = region.shape[:2]
            template_resized = cv2.resize(template, (w, h))
            
            # Template matching
            result = cv2.matchTemplate(region, template_resized, cv2.TM_CCOEFF_NORMED)
            confidence = np.max(result)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match_id = tile_id
        
        return best_match_id, best_confidence
    
    def extract_map(self, map_path = "map.png") -> Optional[List[List[int]]]:
        """Extract map grid from preview image"""
        # debug()
        image_path = self.data_dir / map_path
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        print(f"Processing map {map_path}")
        print(f"Image size: {image.shape}")
        
        # Detect hex grid with debug mode
        hex_cells = self.detect_hex_grid(image, debug_mode=True)
        print(f"Detected {len(hex_cells)} hex cells")
        
        if not hex_cells:
            print("No hex cells detected")
            return None
        
        # Classify tiles
        hex_cells = self.classify_tiles(image, hex_cells)
        
        # Convert to grid format
        grid = self._cells_to_grid(hex_cells)
        
        # Validate against expected data
        self._validate_extraction(grid)
        
        return grid
    
    def _cells_to_grid(self, hex_cells: List[HexCell]) -> List[List[int]]:
        """Convert hex cells to 2D grid format"""
        if not hex_cells:
            return []
        
        # Find grid dimensions
        max_row = max(cell.row for cell in hex_cells)
        max_col = max(cell.col for cell in hex_cells)
        
        # Initialize grid
        grid = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        # Fill grid
        for cell in hex_cells:
            grid[cell.row][cell.col] = cell.tile_id
        
        return grid
    
    def _validate_extraction(self, grid: List[List[int]]):
        """Validate extracted grid against expected data"""
        # Count tiles
        tile_counts = {}
        total_tiles = 0
        
        for row in grid:
            for tile_id in row:
                if tile_id != 0:  # Skip empty tiles
                    tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
                    total_tiles += 1
        
        # Compare with expected distribution if available
    
    def render_map(self, grid: List[List[int]], confidence_grid: Optional[List[List[float]]] = None) -> np.ndarray:
        """Render extracted map for visualization"""
        # Load tile references for renderer
        tile_references = self.renderer.load_tile_references(self.data_dir / "AllTiles")
        
        # Render the grid
        return self.renderer.render_hex_grid(grid, tile_references, confidence_grid, highlight_errors=True)
    
    def generate_validation_report(self, map_path: str, output_dir: str = "outputs"):
        """Generate comprehensive validation report for a map"""
        map_data = {}
        
        print(f"Generating validation report for Map {map_path}")
        
        # Extract the map
        grid = self.extract_map(map_path)
        if not grid:
            print("Failed to extract map")
            return
        
        # Get confidence grid
        confidence_grid = self._get_confidence_grid(map_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get original image path
        image_path = self.data_dir / map_path
        
        # Load tile references
        tile_references = self.renderer.load_tile_references(self.data_dir / "AllTiles")
        
        # Save validation images
        self.renderer.save_validation_images(
            output_path, map_path, image_path, grid, tile_references, confidence_grid
        )
        
        # Generate HTML report
        self._generate_html_report(map_path, map_data, grid, confidence_grid, output_path)
        
        print(f"Validation report generated in {output_path}")
    
    def _get_confidence_grid(self, map_path: int) -> Optional[List[List[float]]]:
        """Get confidence grid for a map (placeholder - would need to store during extraction)"""
        # This would need to be implemented to store confidence values during extraction
        # For now, return None
        return None
    
    def _generate_html_report(self, map_path: int, map_data: dict, grid: List[List[int]], 
                            confidence_grid: Optional[List[List[float]]], output_path: Path):
        """Generate HTML validation report"""
        
        # Calculate statistics
        tile_counts = {}
        total_tiles = 0
        confidence_sum = 0
        confidence_count = 0
        
        for row_idx, row in enumerate(grid):
            for col_idx, tile_id in enumerate(row):
                if tile_id != 0:
                    tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
                    total_tiles += 1
                    
                    if confidence_grid and row_idx < len(confidence_grid) and col_idx < len(confidence_grid[row_idx]):
                        confidence_sum += confidence_grid[row_idx][col_idx]
                        confidence_count += 1
        
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Map {map_path} Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #e0e0e0; padding: 15px; border-radius: 5px; text-align: center; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 800px; border: 1px solid #ccc; }}
                .tile-distribution {{ display: flex; flex-wrap: wrap; gap: 10px; }}
                .tile-item {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Map {map_path} Validation Report</h1>
                <h2>{map_path}</h2>
                <p>Generated validation report for extracted map </p>
            </div>
            
            <div class="section">
                <h3>Summary Statistics</h3>
                <div class="stats">
                    <div class="stat-box">
                        <h4>Avg Confidence</h4>
                        <p>{avg_confidence:.2f}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>Visual Comparison</h3>
                <div class="image-container">
                    <img src="map_{map_path}_comparison.png" alt="Side-by-side comparison">
                    <p>Original vs Extracted Map Comparison</p>
                </div>
            </div>
            
            <div class="section">
                <h3>Rendered Map</h3>
                <div class="image-container">
                    <img src="map_{map_path}_rendered.png" alt="Rendered map">
                    <p>Rendered Map from Extracted Data</p>
                </div>
            </div>
        """
        
        if confidence_grid:
            html_content += f"""
            <div class="section">
                <h3>Confidence Heatmap</h3>
                <div class="image-container">
                    <img src="map_{map_path}_confidence.png" alt="Confidence heatmap">
                    <p>Tile Classification Confidence</p>
                </div>
            </div>
            """
        
        html_content += f"""
            <div class="section">
                <h3>Tile Distribution</h3>
                <div class="tile-distribution">
        """
        
        for tile_id, count in tile_counts.items():
            html_content += f"""
                    <div class="tile-item">
                        <strong>Tile {tile_id}</strong><br>
                        Count: {count}
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_path / f"map_report.html", 'w') as f:
            f.write(html_content)

def main():
    """Main function for testing the map extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WeeWar Map Extractor')
    parser.add_argument('--map-id', type=int, help='Extract specific map ID')
    parser.add_argument('--all', action='store_true', help='Extract all maps')
    parser.add_argument('--validate', action='store_true', help='Generate validation report')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    
    args = parser.parse_args()
    
    extractor = MapExtractor()
    # Extract and render
    grid = extractor.extract_map()
    if grid:
        print(f"\nExtracted grid ({len(grid)}x{len(grid[0]) if grid else 0}):")
        for row in grid:
            print(row)
        
        # Generate validation report
        extractor.generate_validation_report("map.png", args.output_dir)
        print(f"Validation report generated in {args.output_dir}")


if __name__ == "__main__":
    main()
