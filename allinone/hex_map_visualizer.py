import json
import base64
import os
import math
from PIL import Image

class HexMapVisualizer:
    def __init__(self, tiles_folder="../AllTiles"):
        self.tiles_folder = tiles_folder
        self.tile_size = 64  # Default tile size
        self.hex_width = 64
        self.hex_height = 74  # Slightly taller for hex shape
        
    def png_to_base64(self, png_path):
        """Convert PNG file to base64 data URL for embedding in SVG"""
        try:
            with open(png_path, 'rb') as f:
                png_data = f.read()
                base64_data = base64.b64encode(png_data).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        except FileNotFoundError:
            print(f"Warning: Tile file not found: {png_path}")
            return None
    
    def create_hex_clip_path(self):
        """Create SVG clip path for hexagonal tiles"""
        # Calculate hexagon points
        width, height = self.hex_width, self.hex_height
        center_x, center_y = width / 2, height / 2
        
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = center_x + (width / 2 - 2) * math.cos(angle)
            y = center_y + (height / 2 - 2) * math.sin(angle)
            points.append(f"{x:.2f},{y:.2f}")
        
        return f'<clipPath id="hexClip"><polygon points="{" ".join(points)}" /></clipPath>'
    
    def calculate_hex_position(self, row, col):
        """Calculate SVG position for a hexagonal tile"""
        # Hexagonal grid spacing
        h_spacing = self.hex_width * 0.75  # 3/4 of tile width
        v_spacing = self.hex_height * 0.866  # sqrt(3)/2 of tile height
        
        # Calculate position
        x = col * h_spacing
        y = row * v_spacing
        
        # Offset odd rows for hexagonal pattern
        if row % 2 == 1:
            x += h_spacing / 2
        
        return x, y
    
    def generate_svg_hex_map(self, tile_array, output_file="hex_map.svg"):
        """Generate SVG representation of the hex map"""
        if not tile_array:
            print("No tile array provided!")
            return None
        
        # Calculate SVG dimensions
        max_row = len(tile_array)
        max_col = max(len(row) for row in tile_array) if tile_array else 0
        
        # Calculate total SVG size
        h_spacing = self.hex_width * 0.75
        v_spacing = self.hex_height * 0.866
        
        svg_width = (max_col * h_spacing) + (h_spacing / 2) + self.hex_width
        svg_height = (max_row * v_spacing) + self.hex_height
        
        # Start building SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width:.0f}" height="{svg_height:.0f}" xmlns="http://www.w3.org/2000/svg">
<defs>
{self.create_hex_clip_path()}
</defs>
<rect width="100%" height="100%" fill="#f0f0f0"/>
'''
        
        # Add tiles to SVG
        for row_idx, row in enumerate(tile_array):
            for col_idx, tile_id in enumerate(row):
                if tile_id != 0:  # Skip empty tiles
                    x, y = self.calculate_hex_position(row_idx, col_idx)
                    
                    # Get tile image
                    tile_path = os.path.join(self.tiles_folder, f"{tile_id}.png")
                    base64_data = self.png_to_base64(tile_path)
                    
                    if base64_data:
                        svg_content += f'''
<g transform="translate({x:.2f},{y:.2f})">
    <image href="{base64_data}" 
           width="{self.hex_width}" 
           height="{self.hex_height}" 
           clip-path="url(#hexClip)" />
    <text x="{self.hex_width/2}" y="{self.hex_height/2}" 
          text-anchor="middle" 
          dominant-baseline="middle" 
          font-family="Arial" 
          font-size="10" 
          fill="white" 
          stroke="black" 
          stroke-width="0.5">{tile_id}</text>
</g>'''
                    else:
                        # Fallback for missing tiles
                        svg_content += f'''
<g transform="translate({x:.2f},{y:.2f})">
    <polygon points="{" ".join([f"{self.hex_width/2 + (self.hex_width/2-2)*math.cos(i*math.pi/3):.2f},{self.hex_height/2 + (self.hex_height/2-2)*math.sin(i*math.pi/3):.2f}" for i in range(6)])}" 
             fill="gray" 
             stroke="black" 
             stroke-width="1"/>
    <text x="{self.hex_width/2}" y="{self.hex_height/2}" 
          text-anchor="middle" 
          dominant-baseline="middle" 
          font-family="Arial" 
          font-size="12" 
          fill="white">{tile_id}</text>
</g>'''
        
        svg_content += "</svg>"
        
        # Save SVG file
        with open(output_file, 'w') as f:
            f.write(svg_content)
        
        print(f"SVG hex map saved to: {output_file}")
        return output_file
    
    def generate_html_page(self, tile_array, original_image_path="map.png", output_file="hex_map_viewer.html"):
        """Generate HTML page with SVG hex map and original image"""
        # Generate SVG first
        svg_file = self.generate_svg_hex_map(tile_array)
        
        if not svg_file:
            print("Failed to generate SVG map!")
            return None
        
        # Read SVG content
        with open(svg_file, 'r') as f:
            svg_content = f.read()
        
        # Convert original image to base64
        original_base64 = self.png_to_base64(original_image_path)
        if not original_base64:
            print(f"Could not load original image: {original_image_path}")
            return None
        
        # Create HTML page
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hexagonal Map Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .map-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 600px;
        }}
        .map-section h2 {{
            margin-top: 0;
            color: #333;
            text-align: center;
        }}
        .original-image {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .svg-map {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            background: white;
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 5px;
            font-size: 14px;
        }}
        .grid-info {{
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
            background: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1 style="text-align: center; color: #333;">Hexagonal Map Visualization</h1>
    
    <div class="container">
        <div class="map-section">
            <h2>Original Map</h2>
            <img src="{original_base64}" alt="Original Map" class="original-image">
            <div class="info">
                <strong>Original PNG Image:</strong> This is the source hexagonal map image that was analyzed and classified.
            </div>
        </div>
        
        <div class="map-section">
            <h2>Reconstructed SVG Map</h2>
            <div class="svg-map">
                {svg_content}
            </div>
            <div class="info">
                <strong>SVG Reconstruction:</strong> This shows the classified tiles reconstructed using the reference tiles from the AllTiles folder. Each tile is labeled with its ID number.
            </div>
        </div>
    </div>
    
    <div class="map-section" style="margin: 20px auto; max-width: 800px;">
        <h2>Tile Array Data</h2>
        <div class="grid-info">
            <strong>2D Array (0 = empty/hole):</strong><br>
'''
        
        # Add tile array data
        for i, row in enumerate(tile_array):
            html_content += f"Row {i}: {row}<br>"
        
        html_content += '''
        </div>
        <div class="info">
            <strong>Legend:</strong> 
            <ul>
                <li><strong>0</strong> = Empty space or hole in the hexagonal map</li>
                <li><strong>Numbers</strong> = Tile IDs corresponding to files in the AllTiles folder</li>
                <li>Odd rows are offset horizontally to create the hexagonal pattern</li>
            </ul>
        </div>
    </div>
    
</body>
</html>'''
        
        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML visualization saved to: {output_file}")
        return output_file

def load_classification_results(json_file="dynamic_hex_classification.json"):
    """Load classification results from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            return data.get('tile_array', [])
    except FileNotFoundError:
        print(f"Classification file not found: {json_file}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {json_file}")
        return None

if __name__ == "__main__":
    # Load classification results
    tile_array = load_classification_results()
    
    if tile_array:
        # Create visualizer
        visualizer = HexMapVisualizer()
        
        # Generate HTML page with both maps
        html_file = visualizer.generate_html_page(tile_array)
        
        if html_file:
            print(f"\\nVisualization complete!")
            print(f"Open {html_file} in your web browser to view the hex map visualization.")
            print(f"The page shows both the original image and the reconstructed SVG map side by side.")
        else:
            print("Failed to generate visualization!")
    else:
        print("No tile array data found. Please run the dynamic hex classifier first to generate dynamic_hex_classification.json")
