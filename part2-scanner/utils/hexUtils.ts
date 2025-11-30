import { HexConfig, TileCoordinates } from '../types';

/**
 * Calculates the center pixel coordinate for a given hex (q, r)
 * using POINTY-TOP geometry.
 */
export const getHexCenter = (q: number, r: number, config: HexConfig): { x: number, y: number } => {
  const { originX, originY, width, height } = config;
  const x = originX + (q + r/2) * width;
  const y = originY + r * (height * 0.75);

  return { x, y };
};

/**
 * Generates grid coordinates converting from a rectangular loop (col, row)
 * into Axial coordinates (q, r).
 */
export const generateGridCoords = (config: HexConfig): TileCoordinates[] => {
  const coords: TileCoordinates[] = [];
  
  for (let row = 0; row < config.gridHeight; row++) {
    for (let col = 0; col < config.gridWidth; col++) {
      // Convert Offset (col, row) to Axial (q, r)
      // Odd-r offset:
      // q = col - (row - (row&1)) / 2
      // r = row
      
      const q = col - Math.floor((row - (row % 2)) / 2);
      const r = row;

      const { x, y } = getHexCenter(q, r, config);
      coords.push({ q, r, x, y });
    }
  }
  return coords;
};

export const getHexPathData = (width: number, height: number): string => {
  const hw = width / 2;
  const hh = height / 2;
  const qtrH = height / 4;
  
  // Pointy Top Hexagon Path (centered at 0,0)
  return `
    M 0 ${-hh}
    L ${hw} ${-qtrH}
    L ${hw} ${qtrH}
    L 0 ${hh}
    L ${-hw} ${qtrH}
    L ${-hw} ${-qtrH}
    Z
  `;
};

// Crop a specific tile from the source image with Hexagonal Clipping
export const cropTile = (
  image: HTMLImageElement | null,
  center: { x: number, y: number },
  width: number,
  height: number
): string | null => {
  // Defensive checks to prevent canvas errors
  if (!image || image.naturalWidth === 0 || image.naturalHeight === 0) return null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) return null;

  const hw = width / 2;
  
  ctx.beginPath();
  // Top Center
  ctx.moveTo(hw, 0);
  // Top Right
  ctx.lineTo(width, height * 0.25);
  // Bottom Right
  ctx.lineTo(width, height * 0.75);
  // Bottom Center
  ctx.lineTo(hw, height);
  // Bottom Left
  ctx.lineTo(0, height * 0.75);
  // Top Left
  ctx.lineTo(0, height * 0.25);
  ctx.closePath();

  // Apply Clip
  ctx.clip();

  // Calculate Source Rect
  const sourceX = center.x - (width / 2);
  const sourceY = center.y - (height / 2);

  try {
      // Draw the slice (content outside path is transparent)
      ctx.drawImage(
        image,
        sourceX, sourceY, width, height, // Source rect
        0, 0, width, height // Dest rect
      );
    
      // Return PNG to preserve transparency of the clipped area
      return canvas.toDataURL('image/png');
  } catch (e) {
      console.error("Failed to crop tile", e);
      return null;
  }
};

// Create an HTMLImageElement from a base64 string (async)
export const loadImageFromBase64 = (base64: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = base64;
    });
};

// Auto-calculate grid dimensions based on image size
export const calculateGridDimensions = (imgWidth: number, imgHeight: number, hexWidth: number, hexHeight: number) => {
    // Width is straight forward
    const gridWidth = Math.ceil(imgWidth / hexWidth) + 1;

    // Height overlaps by 0.75
    const vertSpacing = hexHeight * 0.75;
    const gridHeight = Math.ceil(imgHeight / vertSpacing) + 1;

    return { gridWidth, gridHeight };
};

/**
 * Generate a simple hash from a base64 image string.
 * Used to dedupe training examples - identical tiles get the same hash
 * regardless of their q/r coordinates.
 */
export const hashBase64Image = (base64: string): string => {
    // Use a simple djb2 hash on the base64 content
    // Strip the data URL prefix if present
    const content = base64.includes(',') ? base64.split(',')[1] : base64;

    let hash = 5381;
    for (let i = 0; i < content.length; i++) {
        hash = ((hash << 5) + hash) + content.charCodeAt(i);
        hash = hash & hash; // Convert to 32-bit integer
    }

    // Return as hex string
    return (hash >>> 0).toString(16);
};