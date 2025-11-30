
import React from 'react';
import { LabelStats, TileProperties } from '../types';
import { CloudArrowUpIcon, ArrowDownTrayIcon, ArrowUpTrayIcon, XMarkIcon, PencilIcon, FunnelIcon, PhotoIcon, EyeIcon, TrashIcon } from '@heroicons/react/24/solid';
import { samplePresets, SamplePreset } from '../samplePresets';

interface KnowledgeBasePanelProps {
    labelStats: LabelStats;
    activeFilter: { category: string, label: string } | null;
    onImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
    onImport: (e: React.ChangeEvent<HTMLInputElement>) => void;
    onExport: () => void;
    onDeleteLabel: (category: keyof TileProperties, label: string) => void;
    onRenameLabel: (category: keyof TileProperties, oldLabel: string, newLabel: string) => void;
    onSelectLabel: (category: keyof TileProperties, label: string) => void;
    onLoadSample?: (preset: SamplePreset) => void;
    onViewTrainingData?: () => void;
    onReset?: () => void;
    exampleCount?: number;
    loadedKBName?: string | null;
}

export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({
    labelStats,
    activeFilter,
    onImageUpload,
    onImport,
    onExport,
    onDeleteLabel,
    onRenameLabel,
    onSelectLabel,
    onLoadSample,
    onViewTrainingData,
    onReset,
    exampleCount = 0,
    loadedKBName
}) => {

    const handleRename = (category: keyof TileProperties, label: string) => {
        const newName = prompt(`Rename or Consolidate "${label}" to:`, label);
        if (newName && newName !== label) {
            onRenameLabel(category, label, newName);
        }
    };

    const renderLabelSection = (category: keyof TileProperties, title: string) => {
        const counts = labelStats[category] || {};
        const labels = Object.keys(counts).sort();
        
        return (
            <div className="mb-4">
                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2 flex justify-between items-center px-1">
                    {title}
                    <span className="bg-gray-700 text-gray-400 px-1.5 rounded-full text-[10px] border border-gray-600">{labels.length}</span>
                </h3>
                <div className="flex flex-col gap-1 px-1">
                    {labels.map(l => {
                        const isActive = activeFilter?.category === category && activeFilter?.label === l;
                        return (
                            <div 
                                key={l} 
                                className={`group flex items-center justify-between px-2 py-1.5 rounded text-xs transition-colors border border-transparent cursor-pointer
                                    ${isActive ? 'bg-blue-900/50 border-blue-600 text-blue-200' : 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:border-gray-500'}
                                `}
                                onClick={() => onSelectLabel(category, l)}
                            >
                                <div className="flex items-center gap-2 overflow-hidden">
                                    {isActive && <FunnelIcon className="w-3 h-3 text-blue-400 flex-shrink-0" />}
                                    <span className="truncate" title={l}>{l}</span>
                                    <span className={`text-[9px] px-1 rounded-full ${isActive ? 'bg-blue-800 text-blue-300' : 'bg-gray-800 text-gray-500'}`}>
                                        {counts[l]}
                                    </span>
                                </div>
                                
                                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button 
                                        onClick={(e) => { e.stopPropagation(); handleRename(category, l); }}
                                        className="p-1 hover:bg-gray-500 rounded text-gray-400 hover:text-white"
                                        title="Rename"
                                    >
                                        <PencilIcon className="w-3 h-3" />
                                    </button>
                                    <button 
                                        onClick={(e) => { e.stopPropagation(); if(confirm(`Delete "${l}"? This will remove all ${counts[l]} training examples for this label.`)) onDeleteLabel(category, l); }}
                                        className="p-1 hover:bg-red-900/50 rounded text-gray-500 hover:text-red-400"
                                        title="Delete All Examples"
                                    >
                                        <XMarkIcon className="w-3 h-3" />
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                    {labels.length === 0 && <span className="text-xs text-gray-600 italic px-1">No data yet. Train tiles to see labels here.</span>}
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full bg-gray-800 text-gray-100">
            {/* Image Section */}
            <div className="p-4 border-b border-gray-700 bg-gray-800 shrink-0">
                <h3 className="text-xs font-bold uppercase text-gray-400 mb-3">Image</h3>
                <label className="block w-full cursor-pointer group mb-3">
                    <input type="file" className="hidden" accept="image/*" onChange={onImageUpload} />
                    <div className="border-2 border-dashed border-gray-600 rounded-lg p-3 flex flex-col items-center justify-center text-gray-400 group-hover:border-blue-500 group-hover:text-blue-400 transition-colors bg-gray-900/30">
                        <CloudArrowUpIcon className="w-6 h-6 mb-1" />
                        <span className="text-xs font-medium">Upload Screenshot</span>
                    </div>
                </label>

                {/* Sample Maps Dropdown */}
                {onLoadSample && samplePresets.length > 0 && (
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Or try a sample:</label>
                        <select
                            className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
                            defaultValue=""
                            onChange={(e) => {
                                const preset = samplePresets.find(p => p.id === e.target.value);
                                if (preset) onLoadSample(preset);
                                e.target.value = ''; // Reset to allow re-selection
                            }}
                        >
                            <option value="" disabled>Select a sample map...</option>
                            {samplePresets.map(preset => (
                                <option key={preset.id} value={preset.id}>
                                    {preset.name}
                                </option>
                            ))}
                        </select>
                    </div>
                )}
            </div>

            {/* Knowledge Base Section */}
            <div className="p-4 border-b border-gray-700 bg-gray-800 shrink-0">
                <h3 className="text-xs font-bold uppercase text-gray-400 mb-3">Knowledge Base</h3>

                {/* Loaded KB indicator */}
                {loadedKBName && (
                    <div className="mb-3 px-2 py-1.5 bg-gray-900/50 rounded border border-gray-600 flex items-center justify-between">
                        <span className="text-xs text-gray-300 truncate" title={loadedKBName}>
                            📁 {loadedKBName}
                        </span>
                        <span className="text-[10px] text-gray-500">{exampleCount} examples</span>
                    </div>
                )}

                <div className="grid grid-cols-4 gap-2">
                    <button onClick={onExport} className="flex flex-col items-center justify-center p-2 bg-gray-700 hover:bg-gray-600 rounded border border-gray-600 transition-colors">
                        <ArrowDownTrayIcon className="w-4 h-4 mb-1 text-blue-400"/>
                        <span className="text-[10px]">Export</span>
                    </button>
                    <label className="flex flex-col items-center justify-center p-2 bg-gray-700 hover:bg-gray-600 rounded border border-gray-600 transition-colors cursor-pointer">
                        <input type="file" className="hidden" accept=".json" onChange={onImport} />
                        <ArrowUpTrayIcon className="w-4 h-4 mb-1 text-green-400"/>
                        <span className="text-[10px]">Import</span>
                    </label>
                    {onViewTrainingData && (
                        <button
                            onClick={onViewTrainingData}
                            className="flex flex-col items-center justify-center p-2 bg-gray-700 hover:bg-gray-600 rounded border border-gray-600 transition-colors"
                        >
                            <EyeIcon className="w-4 h-4 mb-1 text-purple-400"/>
                            <span className="text-[10px]">View</span>
                        </button>
                    )}
                    {onReset && (
                        <button
                            onClick={onReset}
                            className="flex flex-col items-center justify-center p-2 bg-gray-700 hover:bg-red-900/50 rounded border border-gray-600 transition-colors"
                        >
                            <TrashIcon className="w-4 h-4 mb-1 text-red-400"/>
                            <span className="text-[10px]">Reset</span>
                        </button>
                    )}
                </div>
            </div>

            <div className="p-2 bg-gray-900/50 text-[10px] text-gray-400 border-b border-gray-700 text-center">
                Click to Filter Results • Pencil to Rename
            </div>

            <div className="flex-1 overflow-y-auto p-3 custom-scrollbar">
               {renderLabelSection('terrain', 'Terrain')}
               {renderLabelSection('unit', 'Units')}
               {renderLabelSection('tileOwner', 'Tile Owners')}
               {renderLabelSection('unitOwner', 'Unit Owners')}
               {renderLabelSection('infrastructure', 'Infrastructure')}
            </div>
        </div>
    );
};
