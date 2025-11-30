
import React from 'react';
import { HexConfig } from '../types';
import { getHexCenter, getHexPathData } from '../utils/hexUtils';

interface GridOverlayProps {
  config: HexConfig;
  imageWidth: number;
  imageHeight: number;
}

export const GridOverlay: React.FC<GridOverlayProps> = ({ config, imageWidth, imageHeight }) => {
  const { gridWidth, gridHeight, width, height, showLabels } = config;

  // Debug logging
  console.log('GridOverlay render:', { gridWidth, gridHeight, width, height, imageWidth, imageHeight });

  const hexes = [];
  const pathData = getHexPathData(width, height);

  for (let row = 0; row < gridHeight; row++) {
    for (let col = 0; col < gridWidth; col++) {
        // Convert loop to Axial just for consistent rendering with logic
        const q = col - Math.floor((row - (row % 2)) / 2);
        const r = row;
        
        const { x, y } = getHexCenter(q, r, config);

        // Only draw if roughly within bounds
        if (x > -width && x < imageWidth + width && y > -height && y < imageHeight + height) {
        hexes.push(
          <g key={`${q}-${r}`} transform={`translate(${x}, ${y})`}>
            {/* Center Dot */}
            <circle r="1" fill="rgba(255, 255, 0, 0.9)" />
            
            {/* Pointy Top Hex Outline */}
            <path
              d={pathData}
              fill="none"
              stroke="rgba(0, 255, 0, 0.4)"
              strokeWidth="1"
            />
            
            {/* Coords - Only render if enabled */}
            {showLabels && (
              <text 
                y="0" 
                fontSize="12" 
                fontWeight="bold"
                fill="rgba(255,255,255,0.9)" 
                textAnchor="middle" 
                dy="4" 
                style={{ textShadow: '0px 1px 3px black', pointerEvents: 'none' }}
              >
                {q},{r}
              </text>
            )}
          </g>
        );
      }
    }
  }

  return (
    <svg 
      width={imageWidth} 
      height={imageHeight} 
      className="absolute top-0 left-0 pointer-events-none overflow-visible"
    >
      {hexes}
    </svg>
  );
};
