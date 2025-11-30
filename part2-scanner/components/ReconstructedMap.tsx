
import React from 'react';
import { AnalysisResult, HexConfig } from '../types';
import { getHexCenter } from '../utils/hexUtils';
import { getTileImage, getCrossingImage, getUnitImage } from '../utils/assetUtils';

interface ReconstructedMapProps {
    results: AnalysisResult[];
    config: HexConfig;
    width: number;
    height: number;
    zoom: number;
}

export const ReconstructedMap: React.FC<ReconstructedMapProps> = ({ results, config, width, height, zoom }) => {
    const { width: hexW, height: hexH } = config;

    return (
        <div 
            className="relative bg-gray-800 overflow-hidden shadow-inner"
            style={{ 
                width: width, 
                height: height,
                transform: `scale(${zoom})`,
                transformOrigin: '0 0'
            }}
        >
            {/* Base Grid Pattern to show empty space */}
            <div className="absolute inset-0 opacity-10" 
                 style={{ 
                     backgroundImage: `radial-gradient(circle, #ffffff 1px, transparent 1px)`, 
                     backgroundSize: '20px 20px' 
                 }} 
            />

            {results.map((res) => {
                if (res.terrain === 'Empty') return null;

                const { x, y } = getHexCenter(res.q, res.r, config);
                const left = x - hexW / 2;
                const top = y - hexH / 2;

                const tileImg = getTileImage(res.terrain, res.tileOwner);
                const crossingImg = getCrossingImage(res.infrastructure);
                const unitImg = getUnitImage(res.unit, res.unitOwner);

                return (
                    <div 
                        key={`${res.q},${res.r}`}
                        className="absolute pointer-events-none flex items-center justify-center"
                        style={{ 
                            left: left, 
                            top: top, 
                            width: hexW, 
                            height: hexH 
                        }}
                    >
                        {/* 1. Terrain Layer */}
                        {tileImg && <img src={tileImg} alt="Terrain" className="absolute inset-0 w-full h-full object-contain scale-110" />}
                        
                        {/* 2. Infrastructure Layer */}
                        {crossingImg && <img src={crossingImg} alt="Infra" className="absolute inset-0 w-full h-full object-contain z-10" />}
                        
                        {/* 3. Unit Layer */}
                        {unitImg && <img src={unitImg} alt="Unit" className="absolute inset-0 w-full h-full object-contain z-20 scale-75" />}
                    </div>
                );
            })}
        </div>
    );
};
