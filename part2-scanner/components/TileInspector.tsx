
import React, { useState, useEffect } from 'react';
import { InspectorData, TileProperties } from '../types';
import { XMarkIcon, CheckIcon, SparklesIcon } from '@heroicons/react/24/solid';

interface TileInspectorProps {
    tile: InspectorData | null;
    activeLabels: {
        terrain: string[];
        unit: string[];
        tileOwner: string[];
        unitOwner: string[];
        infrastructure: string[];
    };
    onSave: (props: TileProperties) => void;
    onClose: () => void;
}

const DEFAULT_PROPS: TileProperties = {
    terrain: 'Grass',
    unit: 'None',
    tileOwner: 'None',
    unitOwner: 'None',
    infrastructure: 'None'
};

const SUGGESTED_LABELS: Record<keyof TileProperties, string[]> = {
    terrain: ['Empty', 'Grass', 'Water', 'Forest', 'Mountain', 'Shoal', 'Reef', 'Plain', 'Sea'],
    unit: ['None', 'Infantry', 'Mech', 'Tank', 'Artillery', 'Rocket', 'Anti-Air', 'Missile', 'Fighter', 'Bomber', 'B-Copter', 'T-Copter', 'Battleship', 'Cruiser', 'Lander', 'Sub'],
    tileOwner: ['None', 'Neutral', 'Red', 'Blue', 'Green', 'Yellow', 'Black'],
    unitOwner: ['None', 'Red', 'Blue', 'Green', 'Yellow', 'Black'],
    infrastructure: ['None', 'Road', 'Bridge', 'River', 'City', 'Base', 'Airport', 'Port', 'HQ']
};

export const TileInspector: React.FC<TileInspectorProps> = ({ tile, activeLabels, onSave, onClose }) => {
    const [props, setProps] = useState<TileProperties>(DEFAULT_PROPS);
    
    // Reset when tile changes
    useEffect(() => {
        if (tile) {
            // Priority: Existing Label > Predicted Label > Default
            const initial = tile.existingLabels || tile.predictedLabels || DEFAULT_PROPS;
            setProps({
                terrain: initial.terrain || 'None',
                unit: initial.unit || 'None',
                tileOwner: initial.tileOwner || 'None',
                unitOwner: initial.unitOwner || 'None',
                infrastructure: initial.infrastructure || 'None'
            });
        }
    }, [tile]);

    if (!tile) return null;

    const handleChange = (cat: keyof TileProperties, value: string) => {
        setProps(prev => ({ ...prev, [cat]: value }));
    };

    const renderInput = (cat: keyof TileProperties, label: string) => {
        // Check if current input matches prediction
        const prediction = tile.predictedLabels?.[cat];
        const showDiff = prediction && prediction !== props[cat] && prediction !== 'undefined' && prediction !== 'None';
        
        // Merge active labels (from dataset) with suggestions, deduped
        const options = Array.from(new Set([
            ...(activeLabels[cat] || []), 
            ...SUGGESTED_LABELS[cat]
        ])).sort();
        
        // Ensure value isn't literally "undefined" string or empty
        const safeValue = (props[cat] === 'undefined' || !props[cat]) ? 'None' : props[cat];

        return (
            <div className="space-y-1">
                <div className="flex justify-between items-center">
                    <label className="text-xs font-bold text-gray-400 uppercase">{label}</label>
                    {showDiff && (
                        <span className="text-[10px] bg-blue-900/50 text-blue-300 px-1.5 py-0.5 rounded border border-blue-800 flex items-center gap-1" title="AI Predicted this value">
                            <SparklesIcon className="w-2.5 h-2.5" />
                            AI: {prediction}
                        </span>
                    )}
                </div>
                <div className="relative">
                    <input 
                        type="text"
                        list={`list-${cat}`}
                        value={safeValue} 
                        onChange={e => handleChange(cat, e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm text-white focus:border-blue-500 outline-none placeholder-gray-600"
                        placeholder={`Select or type ${label}...`}
                        onFocus={(e) => e.target.select()}
                    />
                    <datalist id={`list-${cat}`}>
                        {options.map(l => <option key={l} value={l} />)}
                    </datalist>
                </div>
            </div>
        );
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <div className="bg-gray-800 rounded-xl shadow-2xl border border-gray-700 w-full max-w-sm flex flex-col overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="bg-gray-900 p-3 border-b border-gray-700 flex justify-between items-center">
                    <h3 className="font-bold text-white flex items-center gap-2">
                        Inspect Tile <span className="text-xs bg-gray-700 px-2 py-0.5 rounded font-mono text-gray-300">Q:{tile.q} R:{tile.r}</span>
                    </h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">
                        <XMarkIcon className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-4 space-y-4">
                    {/* Image Preview */}
                    <div className="flex justify-center bg-black/40 rounded-lg p-4 border border-gray-700 border-dashed relative">
                        <img 
                            src={tile.base64} 
                            alt="Tile" 
                            className="w-24 h-24 object-contain rendering-pixelated shadow-lg"
                            style={{ imageRendering: 'pixelated' }}
                        />
                        {tile.existingLabels && (
                            <div className="absolute top-2 right-2">
                                <span className="text-[10px] bg-green-900 text-green-300 px-2 py-0.5 rounded border border-green-700">Saved</span>
                            </div>
                        )}
                    </div>

                    {/* Form */}
                    <div className="grid grid-cols-2 gap-4">
                        {renderInput('terrain', 'Terrain')}
                        {renderInput('unit', 'Unit')}
                        {renderInput('tileOwner', 'Tile Owner')}
                        {renderInput('unitOwner', 'Unit Owner')}
                        <div className="col-span-2">
                            {renderInput('infrastructure', 'Crossing/Road')}
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 bg-gray-900/50 border-t border-gray-700 flex gap-2">
                    <button 
                        onClick={onClose}
                        className="flex-1 px-4 py-2 text-sm font-medium text-gray-400 hover:bg-gray-700 rounded transition-colors"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={() => onSave(props)}
                        className="flex-1 px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white rounded shadow transition-colors flex justify-center items-center gap-2"
                    >
                        <CheckIcon className="w-4 h-4" /> Save & Train
                    </button>
                </div>
            </div>
        </div>
    );
};
