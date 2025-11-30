
import { TileProperties } from "../types";

const COLORS: Record<string, string> = {
    Red: '#ef4444',
    Blue: '#3b82f6',
    Green: '#22c55e',
    Yellow: '#eab308',
    Black: '#1f2937',
    Neutral: '#d1d5db',
    None: '#9ca3af'
};

const TERRAIN_COLORS: Record<string, string> = {
    Grass: '#86efac',     // green-300
    Forest: '#15803d',    // green-700
    Mountain: '#78716c',  // stone-500
    Water: '#60a5fa',     // blue-400
    Sea: '#3b82f6',       // blue-500
    Shoal: '#fef08a',     // yellow-200
    Reef: '#1e40af',      // blue-800
    Plain: '#bef264',     // lime-300
    City: '#e5e7eb',      // gray-200
    Base: '#9ca3af',      // gray-400
    Airport: '#9ca3af',
    Port: '#9ca3af',
    HQ: '#4b5563',
    Empty: 'transparent'
};

const svgToDataUrl = (svg: string) => `data:image/svg+xml;base64,${btoa(svg)}`;

export const getTileImage = (tileType: string, tilePlayer: string): string => {
    if (!tileType || tileType === 'None' || tileType === 'Empty') return '';

    // Base color from terrain
    let fill = TERRAIN_COLORS[tileType] || '#cbd5e1'; // default slate-300

    // If it's a property (City, Base, etc), tint it with player color
    const isProperty = ['City', 'Base', 'Airport', 'Port', 'HQ'].includes(tileType);
    let stroke = 'none';
    let strokeWidth = '0';

    if (isProperty) {
        const playerColor = COLORS[tilePlayer] || COLORS.Neutral;
        // For properties, we might use the player color as the main fill or a strong border
        // Let's use player color as border/accent
        stroke = playerColor;
        strokeWidth = '8';
        // Or replace fill for HQ
        if (tileType === 'HQ') fill = playerColor;
    }

    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <polygon points="50 0, 100 25, 100 75, 50 100, 0 75, 0 25" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" />
        ${tileType === 'Forest' ? '<circle cx="30" cy="40" r="15" fill="#14532d" opacity="0.6"/><circle cx="70" cy="40" r="15" fill="#14532d" opacity="0.6"/><circle cx="50" cy="70" r="20" fill="#14532d" opacity="0.8"/>' : ''}
        ${tileType === 'Mountain' ? '<path d="M20 80 L50 20 L80 80 Z" fill="#57534e" opacity="0.5"/>' : ''}
    </svg>`;

    return svgToDataUrl(svg);
};

export const getCrossingImage = (crossingType: string): string => {
    if (!crossingType || crossingType === 'None') return '';

    const color = crossingType === 'River' ? '#3b82f6' : '#d1d5db'; // Blue for river, Gray for road
    
    // Simple representation: A line or cross
    // Road usually goes through center. Let's draw a generic "cross" shape for Road/River
    // In a real app, this needs adjacency logic (Road from Top-Left to Bottom-Right etc)
    // Here we just draw a generic icon.
    
    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path d="M50 0 L50 100 M0 50 L100 50" stroke="${color}" stroke-width="15" fill="none" />
        ${crossingType === 'Bridge' ? '<rect x="30" y="35" width="40" height="30" fill="#78350f" />' : ''}
    </svg>`;

    return svgToDataUrl(svg);
};

export const getUnitImage = (unitType: string, unitPlayer: string): string => {
    if (!unitType || unitType === 'None') return '';

    const color = COLORS[unitPlayer] || '#000000';
    
    // Simple icon generation based on first letter or specific shapes
    let shape = `<rect x="25" y="25" width="50" height="50" fill="${color}" />`;
    
    // Distinguish a few types
    if (unitType.includes('Infantry') || unitType.includes('Mech')) {
        // Circle (Head) + Body
        shape = `<circle cx="50" cy="40" r="15" fill="${color}" /><rect x="35" y="55" width="30" height="30" fill="${color}" />`;
    } else if (unitType.includes('Copter') || unitType.includes('Air')) {
        // Triangle
        shape = `<polygon points="50 20, 80 80, 20 80" fill="${color}" />`;
    } else if (unitType.includes('Ship') || unitType.includes('Lander')) {
        // Boat shape
        shape = `<path d="M20 60 L80 60 L70 90 L30 90 Z" fill="${color}" /><rect x="45" y="30" width="10" height="30" fill="${color}" />`;
    } else {
        // Tank / Default (Boxy)
        shape = `<rect x="20" y="35" width="60" height="40" rx="5" fill="${color}" /><rect x="40" y="20" width="20" height="20" fill="${color}" />`;
    }

    const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="white" fill-opacity="0.5" stroke="white" stroke-width="2"/>
        <g transform="scale(0.8) translate(12,12)">
            ${shape}
        </g>
        <text x="50" y="95" font-family="sans-serif" font-size="20" text-anchor="middle" fill="white" stroke="black" stroke-width="0.5" font-weight="bold">${unitType}</text>
    </svg>`;

    return svgToDataUrl(svg);
};
