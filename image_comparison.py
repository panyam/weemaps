import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
import os
import math
from scipy import fftpack

class ImageComparator:
    def __init__(self):
        self.hash_size = 8  # Standard for p-hash
    
    def svg_to_png(self, svg_file, png_file, width=None, height=None):
        """Convert SVG to PNG using simple rendering approach"""
        try:
            # For this implementation, we'll render the SVG by recreating it as PNG
            # This is a simplified approach - in production you'd use cairosvg or similar
            
            # Load the classification results to recreate the image
            tile_array = self.load_classification_results()
            if not tile_array:
                print("Could not load tile array for SVG conversion")
                return False
            
            # Create PNG version of the hex map
            return self.create_png_from_tiles(tile_array, png_file, width, height)
            
        except Exception as e:
            print(f"Error converting SVG to PNG: {e}")
            return False
    
    def load_classification_results(self, json_file="improved_hex_classification.json"):
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
    
    def create_png_from_tiles(self, tile_array, output_file, target_width=None, target_height=None):
        """Create PNG image from tile array"""
        if not tile_array:
            return False
        
        # Load a reference tile to get dimensions
        tiles_folder = "AllTiles"
        ref_tile_path = None
        for filename in os.listdir(tiles_folder):
            if filename.endswith('.png'):
                ref_tile_path = os.path.join(tiles_folder, filename)
                break
        
        if not ref_tile_path:
            print("No reference tiles found")
            return False
        
        ref_img = cv2.imread(ref_tile_path)
        if ref_img is None:
            print(f"Could not load reference tile: {ref_tile_path}")
            return False
        
        tile_height, tile_width = ref_img.shape[:2]
        
        # Calculate canvas dimensions
        max_row = len(tile_array)
        max_col = max(len(row) for row in tile_array) if tile_array else 0
        
        # Hexagonal spacing
        h_spacing = int(tile_width * 0.75)
        v_spacing = int(tile_height * 0.866)
        
        canvas_width = int((max_col * h_spacing) + (h_spacing / 2) + tile_width)
        canvas_height = int((max_row * v_spacing) + tile_height)
        
        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas.fill(240)  # Light gray background
        
        # Place tiles on canvas
        for row_idx, row in enumerate(tile_array):
            for col_idx, tile_id in enumerate(row):
                if tile_id != 0:  # Skip empty tiles
                    # Calculate position
                    x = int(col_idx * h_spacing)
                    y = int(row_idx * v_spacing)
                    
                    # Offset odd rows
                    if row_idx % 2 == 1:
                        x += int(h_spacing / 2)
                    
                    # Load tile image
                    tile_path = os.path.join(tiles_folder, f"{tile_id}.png")
                    if os.path.exists(tile_path):
                        tile_img = cv2.imread(tile_path)
                        if tile_img is not None:
                            # Place tile on canvas
                            end_y = min(y + tile_height, canvas_height)
                            end_x = min(x + tile_width, canvas_width)
                            
                            if y < canvas_height and x < canvas_width:
                                canvas[y:end_y, x:end_x] = tile_img[0:end_y-y, 0:end_x-x]
        
        # Resize to target dimensions if specified
        if target_width and target_height:
            canvas = cv2.resize(canvas, (target_width, target_height))
        
        # Save image
        cv2.imwrite(output_file, canvas)
        print(f"Generated PNG from tiles: {output_file}")
        return True
    
    def calculate_phash(self, image_path):
        """Calculate perceptual hash of an image using manual implementation"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to 32x32 for p-hash calculation
            resized = cv2.resize(gray, (32, 32))
            
            # Apply DCT (Discrete Cosine Transform)
            dct = fftpack.dct(fftpack.dct(resized.T, norm='ortho').T, norm='ortho')
            
            # Extract top-left 8x8 corner (low frequency components)
            dct_low = dct[:self.hash_size, :self.hash_size]
            
            # Calculate median of DCT coefficients
            median = np.median(dct_low)
            
            # Create binary hash based on median
            hash_bits = dct_low > median
            
            # Convert to integer hash
            hash_value = 0
            for i in range(self.hash_size):
                for j in range(self.hash_size):
                    if hash_bits[i, j]:
                        hash_value |= (1 << (i * self.hash_size + j))
            
            return hash_value
        except Exception as e:
            print(f"Error calculating p-hash for {image_path}: {e}")
            return None
    
    def calculate_similarity(self, hash1, hash2):
        """Calculate similarity between two hashes (0-100%)"""
        if hash1 is None or hash2 is None:
            return 0
        
        # Calculate Hamming distance (number of different bits)
        xor_result = hash1 ^ hash2
        hamming_distance = bin(xor_result).count('1')
        
        # Convert to similarity percentage
        max_distance = self.hash_size * self.hash_size
        similarity = (1 - (hamming_distance / max_distance)) * 100
        
        return max(0, similarity)
    
    def calculate_structural_similarity(self, img1_path, img2_path):
        """Calculate structural similarity (SSIM) between two images"""
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return 0
            
            # Resize images to same size for comparison
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(gray1, gray2)
            
            return ssim * 100
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0
    
    def create_difference_image(self, img1_path, img2_path, output_path):
        """Create a visual difference image highlighting differences"""
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return False
            
            # Resize to same dimensions
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))
            
            # Calculate absolute difference
            diff = cv2.absdiff(img1_resized, img2_resized)
            
            # Enhance the difference for visualization
            diff_enhanced = cv2.convertScaleAbs(diff, alpha=3, beta=0)
            
            # Create a colored difference image
            diff_colored = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
            
            # Save difference image
            cv2.imwrite(output_path, diff_colored)
            print(f"Difference image saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating difference image: {e}")
            return False
    
    def compare_images(self, original_path, generated_svg_path=None, generated_png_path=None):
        """Compare original image with generated map"""
        print("=== Image Comparison Report ===")
        print(f"Original image: {original_path}")
        
        # Use the correct generated map
        if generated_png_path is None:
            generated_png_path = "correct_generated_map.png"
            
        # Check if correct map exists
        if not os.path.exists(generated_png_path):
            print(f"Correct map not found: {generated_png_path}")
            print("Run correct_hex_generator.py first!")
            return None
        
        print(f"Generated PNG: {generated_png_path}")
        
        # Calculate p-hashes
        original_hash = self.calculate_phash(original_path)
        generated_hash = self.calculate_phash(generated_png_path)
        
        print(f"Original p-hash: {original_hash}")
        print(f"Generated p-hash: {generated_hash}")
        
        # Calculate similarity metrics
        phash_similarity = self.calculate_similarity(original_hash, generated_hash)
        ssim_similarity = self.calculate_structural_similarity(original_path, generated_png_path)
        hamming_distance = bin(original_hash ^ generated_hash).count('1') if original_hash and generated_hash else None
        
        print(f"\\nSimilarity Metrics:")
        print(f"P-Hash Similarity: {phash_similarity:.2f}%")
        print(f"SSIM Similarity: {ssim_similarity:.2f}%")
        print(f"Hamming Distance: {hamming_distance} bits")
        
        # Create difference image
        diff_path = "difference_map.png"
        if self.create_difference_image(original_path, generated_png_path, diff_path):
            print(f"Difference visualization: {diff_path}")
        
        # Generate detailed report
        report = {
            'original_image': original_path,
            'generated_image': generated_png_path,
            'original_hash': original_hash,
            'generated_hash': generated_hash,
            'phash_similarity': phash_similarity,
            'ssim_similarity': ssim_similarity,
            'hamming_distance': hamming_distance,
            'difference_image': diff_path
        }
        
        # Save report
        with open('comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\nDetailed report saved: comparison_report.json")
        
        return report

if __name__ == "__main__":
    comparator = ImageComparator()
    
    # Compare original with generated map
    report = comparator.compare_images("map.png")
    
    if report:
        print(f"\\n=== Summary ===")
        print(f"P-Hash Similarity: {report['phash_similarity']:.2f}%")
        print(f"SSIM Similarity: {report['ssim_similarity']:.2f}%")
        
        if report['phash_similarity'] > 80:
            print("✅ High similarity - Generated map closely matches original")
        elif report['phash_similarity'] > 60:
            print("⚠️  Medium similarity - Generated map has some differences")
        else:
            print("❌ Low similarity - Generated map significantly differs from original")
    else:
        print("Comparison failed!")