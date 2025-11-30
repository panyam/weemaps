import React, { useState, useMemo } from 'react';
import { TrainingExample, TileProperties, LabelStats } from '../types';
import { XMarkIcon, PencilIcon, TrashIcon, FunnelIcon } from '@heroicons/react/24/solid';

interface FilterState {
  terrain: { enabled: boolean; value: string };
  unit: { enabled: boolean; value: string };
  tileOwner: { enabled: boolean; value: string };
  unitOwner: { enabled: boolean; value: string };
  infrastructure: { enabled: boolean; value: string };
}

interface TrainingDataViewerProps {
  examples: TrainingExample[];
  labelStats: LabelStats;
  onClose: () => void;
  onEdit: (example: TrainingExample) => void;
  onDelete: (id: string) => void;
}

const CATEGORY_LABELS: Record<keyof TileProperties, string> = {
  terrain: 'Terrain',
  unit: 'Unit',
  tileOwner: 'Tile Owner',
  unitOwner: 'Unit Owner',
  infrastructure: 'Infrastructure',
};

export const TrainingDataViewer: React.FC<TrainingDataViewerProps> = ({
  examples,
  labelStats,
  onClose,
  onEdit,
  onDelete,
}) => {
  const [filters, setFilters] = useState<FilterState>({
    terrain: { enabled: false, value: '' },
    unit: { enabled: false, value: '' },
    tileOwner: { enabled: false, value: '' },
    unitOwner: { enabled: false, value: '' },
    infrastructure: { enabled: false, value: '' },
  });

  const [selectedExample, setSelectedExample] = useState<TrainingExample | null>(null);

  // Get unique labels for each category from the examples
  const availableLabels = useMemo(() => {
    const labels: Record<keyof TileProperties, string[]> = {
      terrain: [],
      unit: [],
      tileOwner: [],
      unitOwner: [],
      infrastructure: [],
    };

    examples.forEach((ex) => {
      (Object.keys(labels) as Array<keyof TileProperties>).forEach((cat) => {
        const val = ex.labels[cat];
        if (val && val !== 'None' && val !== 'undefined' && !labels[cat].includes(val)) {
          labels[cat].push(val);
        }
      });
    });

    // Sort each category
    Object.keys(labels).forEach((cat) => {
      labels[cat as keyof TileProperties].sort();
    });

    return labels;
  }, [examples]);

  // Filter examples based on active filters
  const filteredExamples = useMemo(() => {
    return examples.filter((ex) => {
      for (const cat of Object.keys(filters) as Array<keyof TileProperties>) {
        const filter = filters[cat];
        if (filter.enabled && filter.value) {
          if (ex.labels[cat] !== filter.value) {
            return false;
          }
        }
      }
      return true;
    });
  }, [examples, filters]);

  const toggleFilter = (category: keyof TileProperties) => {
    setFilters((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        enabled: !prev[category].enabled,
        value: !prev[category].enabled ? (availableLabels[category][0] || '') : '',
      },
    }));
  };

  const setFilterValue = (category: keyof TileProperties, value: string) => {
    setFilters((prev) => ({
      ...prev,
      [category]: { ...prev[category], value },
    }));
  };

  const clearAllFilters = () => {
    setFilters({
      terrain: { enabled: false, value: '' },
      unit: { enabled: false, value: '' },
      tileOwner: { enabled: false, value: '' },
      unitOwner: { enabled: false, value: '' },
      infrastructure: { enabled: false, value: '' },
    });
  };

  const activeFilterCount = Object.values(filters).filter((f) => f.enabled).length;

  const handleDelete = (id: string | undefined) => {
    if (!id) return;
    if (confirm('Delete this training example?')) {
      onDelete(id);
      setSelectedExample(null);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">Training Data</h2>
            <span className="text-sm text-gray-400">
              {filteredExamples.length} of {examples.length} examples
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            <XMarkIcon className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Filters */}
        <div className="p-4 border-b border-gray-700 bg-gray-900/50">
          <div className="flex items-center gap-2 mb-3">
            <FunnelIcon className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Filter by features:</span>
            {activeFilterCount > 0 && (
              <button
                onClick={clearAllFilters}
                className="text-xs text-blue-400 hover:text-blue-300 ml-2"
              >
                Clear all
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {(Object.keys(filters) as Array<keyof TileProperties>).map((cat) => (
              <div key={cat} className="flex flex-col gap-1">
                <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters[cat].enabled}
                    onChange={() => toggleFilter(cat)}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
                  />
                  {CATEGORY_LABELS[cat]}
                </label>
                {filters[cat].enabled && (
                  <select
                    value={filters[cat].value}
                    onChange={(e) => setFilterValue(cat, e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
                  >
                    <option value="">Any</option>
                    {availableLabels[cat].map((label) => (
                      <option key={label} value={label}>
                        {label} ({labelStats[cat][label] || 0})
                      </option>
                    ))}
                  </select>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Examples Grid */}
        <div className="flex-1 overflow-auto p-4">
          {filteredExamples.length === 0 ? (
            <div className="text-center text-gray-500 py-12">
              {examples.length === 0
                ? 'No training examples yet. Train some tiles first!'
                : 'No examples match the current filters.'}
            </div>
          ) : (
            <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-2">
              {filteredExamples.map((ex) => (
                <div
                  key={ex.id || ex.timestamp}
                  className={`relative group cursor-pointer rounded overflow-hidden border-2 transition-all ${
                    selectedExample?.id === ex.id
                      ? 'border-blue-500 ring-2 ring-blue-500/50'
                      : 'border-transparent hover:border-gray-500'
                  }`}
                  onClick={() => setSelectedExample(ex)}
                >
                  <img
                    src={ex.imageBase64}
                    alt={ex.labels.terrain}
                    className="w-full aspect-square object-cover bg-gray-900"
                  />
                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onEdit(ex);
                      }}
                      className="p-1 bg-blue-600 rounded hover:bg-blue-500"
                      title="Edit"
                    >
                      <PencilIcon className="w-3 h-3 text-white" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(ex.id);
                      }}
                      className="p-1 bg-red-600 rounded hover:bg-red-500"
                      title="Delete"
                    >
                      <TrashIcon className="w-3 h-3 text-white" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Selected Example Details */}
        {selectedExample && (
          <div className="p-4 border-t border-gray-700 bg-gray-900/50">
            <div className="flex items-start gap-4">
              <img
                src={selectedExample.imageBase64}
                alt="Selected tile"
                className="w-16 h-16 rounded border border-gray-600 object-cover"
              />
              <div className="flex-1 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                {(Object.keys(CATEGORY_LABELS) as Array<keyof TileProperties>).map((cat) => (
                  <div key={cat}>
                    <span className="text-gray-500">{CATEGORY_LABELS[cat]}:</span>
                    <span className="ml-1 text-gray-200">{selectedExample.labels[cat] || 'None'}</span>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => onEdit(selectedExample)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded flex items-center gap-1"
                >
                  <PencilIcon className="w-3 h-3" /> Edit
                </button>
                <button
                  onClick={() => handleDelete(selectedExample.id)}
                  className="px-3 py-1 bg-red-600 hover:bg-red-500 text-white text-sm rounded flex items-center gap-1"
                >
                  <TrashIcon className="w-3 h-3" /> Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
