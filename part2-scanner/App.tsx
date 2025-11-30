import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { HexConfig, ProcessingStatus, AnalysisResult, InspectorData, TileProperties, KnowledgeBase, LabelStats } from './types';
import { GridOverlay } from './components/GridOverlay';
import { TileInspector } from './components/TileInspector';
import { KnowledgeBasePanel } from './components/KnowledgeBasePanel';
import { ReconstructedMap } from './components/ReconstructedMap';
import { cropTile, getHexCenter, loadImageFromBase64, generateGridCoords, calculateGridDimensions, getHexPathData, hashBase64Image } from './utils/hexUtils';
import { loadModels, trainTile, predictTile, importDataset, getDataset, renameLabelInDataset, deleteLabelFromDataset, getExampleById, getLabelCounts, getAllExamples, deleteExampleById, updateExampleLabels, resetDataset } from './services/tfService';
import { TrainingDataViewer } from './components/TrainingDataViewer';
import { TrainingExample } from './types';
import { detectGrid } from './utils/autoCalibration';
import { AdjustmentsHorizontalIcon, PlayIcon, StopIcon, MagnifyingGlassPlusIcon, SparklesIcon, EyeIcon, Square2StackIcon, CloudArrowUpIcon, PhotoIcon } from '@heroicons/react/24/solid';
import { samplePresets, SamplePreset, defaultKBPath } from './samplePresets';

// Debounced Input Component to handle floating point inputs without cursor jumping
const DebouncedNumberInput = ({ label, value, onChange }: { label: string, value: number, onChange: (val: number) => void }) => {
    const [localValue, setLocalValue] = useState(value.toString());
    const [timer, setTimer] = useState<ReturnType<typeof setTimeout> | null>(null);

    // Sync prop changes to local state
    // We only update local state if the incoming prop is numerically different from current local value.
    // This allows "10." to remain "10." even if parent sends back 10.
    useEffect(() => {
        const parsed = parseFloat(localValue);
        // Use a small epsilon for float comparison or just check isNaN
        if (isNaN(parsed) || Math.abs(parsed - value) > 0.0001) {
            setLocalValue(value.toString());
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [value]); 

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value;
        setLocalValue(val);

        if (timer) clearTimeout(timer);

        const newTimer = setTimeout(() => {
            const num = parseFloat(val);
            if (!isNaN(num)) {
                onChange(num);
            }
        }, 500); // 500ms debounce
        setTimer(newTimer);
    };

    return (
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>{label}</span>
          </div>
          <input 
              type="number" 
              step="0.1"
              value={localValue} 
              onChange={handleChange} 
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none" 
          />
      </div>
    );
};

const App: React.FC = () => {
  // --- State ---
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement>(null);
  
  // Refs for scrolling containers
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const reconContainerRef = useRef<HTMLDivElement>(null);
  const isSyncingScroll = useRef(false);

  const [status, setStatus] = useState<ProcessingStatus>(ProcessingStatus.IDLE);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [progress, setProgress] = useState(0);

  // View Mode: Original, Split, Reconstructed
  const [viewMode, setViewMode] = useState<'original' | 'split' | 'reconstructed'>('original');

  // Training Data Viewer
  const [showTrainingViewer, setShowTrainingViewer] = useState(false);
  const [trainingExamples, setTrainingExamples] = useState<TrainingExample[]>([]);

  // Loaded Knowledge Base filename
  const [loadedKBName, setLoadedKBName] = useState<string | null>(null);

  // Visual scaling (does not affect grid logic, just display)
  const [zoom, setZoom] = useState(1.0);

  // Hex Config with LocalStorage persistence
  const [hexConfig, setHexConfig] = useState<HexConfig>(() => {
      try {
          const saved = localStorage.getItem('hexMapConfig');
          if (saved) {
              const parsed = JSON.parse(saved);
              // Ensure legacy configs have showLabels
              if (parsed.showLabels === undefined) parsed.showLabels = true;
              return parsed;
          }
      } catch (e) {
          console.error("Failed to load hex config", e);
      }
      return {
        originX: 32,
        originY: 32,
        width: 64,  
        height: 72, 
        gridWidth: 10,
        gridHeight: 10,
        showLabels: true
      };
  });

  // Save config on change
  useEffect(() => {
      localStorage.setItem('hexMapConfig', JSON.stringify(hexConfig));
  }, [hexConfig]);

  // Knowledge Base State
  const [labelStats, setLabelStats] = useState<LabelStats>({
      terrain: {}, unit: {}, tileOwner: {}, unitOwner: {}, infrastructure: {}
  });
  const [activeLabelFilter, setActiveLabelFilter] = useState<{category: string, label: string} | null>(null);

  const [isModelReady, setIsModelReady] = useState(false);
  const [inspectorTile, setInspectorTile] = useState<InspectorData | null>(null);
  const [highlightedTile, setHighlightedTile] = useState<{q: number, r: number} | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // --- Initialization ---
  useEffect(() => {
    const initTF = async () => {
        try {
            setStatus(ProcessingStatus.LOADING_MODEL);
            await loadModels();
            setIsModelReady(true);

            // Load default KB if configured
            if (defaultKBPath) {
                try {
                    const response = await fetch(defaultKBPath);
                    if (response.ok) {
                        const kb = await response.json() as KnowledgeBase;
                        if (kb.examples && kb.examples.length > 0) {
                            await importDataset(kb);
                            const filename = defaultKBPath.split('/').pop() || 'default-kb.json';
                            setLoadedKBName(filename);
                            console.log(`Loaded default KB: ${filename} (${kb.examples.length} examples)`);
                        }
                    }
                } catch (kbErr) {
                    console.log('No default KB found or failed to load:', kbErr);
                }
            }

            setLabelStats(getLabelCounts());
            setStatus(ProcessingStatus.IDLE);
        } catch (e) {
            console.error("Failed to load models", e);
            setStatus(ProcessingStatus.ERROR);
        }
    };
    setTimeout(initTF, 1000);
  }, []);

  // --- Derived State for Results (Sorted) ---
  const displayResults = useMemo(() => {
      return results
        .filter(r => r.terrain !== 'Empty')
        .sort((a, b) => {
            if (a.q !== b.q) return a.q - b.q;
            return a.r - b.r;
        });
  }, [results]);

  // Derived active labels list for the Inspector (keys of stats)
  const activeLabelsForInspector = useMemo(() => {
      return {
          terrain: Object.keys(labelStats.terrain),
          unit: Object.keys(labelStats.unit),
          tileOwner: Object.keys(labelStats.tileOwner),
          unitOwner: Object.keys(labelStats.unitOwner),
          infrastructure: Object.keys(labelStats.infrastructure)
      };
  }, [labelStats]);

  // --- Handlers ---

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setImageSrc(url);
        setImageDims({ width: img.width, height: img.height });
        setResults([]);
        setHighlightedTile(null);
        setZoom(1.0); // Reset zoom on new image

        const { gridWidth, gridHeight } = calculateGridDimensions(img.width, img.height, hexConfig.width, hexConfig.height);
        setHexConfig(prev => ({ ...prev, gridWidth, gridHeight }));
      };
      img.src = url;
    }
  };

  const loadSamplePreset = (preset: SamplePreset) => {
    const img = new Image();
    img.onload = () => {
      // Clear image first to ensure React sees a state change
      setImageSrc(null);
      setImageDims({ width: 0, height: 0 });

      // Then set the new values in the next tick to force re-render
      setTimeout(() => {
        setImageSrc(img.src);
        setImageDims({ width: img.width, height: img.height });
        setResults([]);
        setHighlightedTile(null);
        setZoom(1.0);
        setHexConfig(prev => ({
          ...prev,
          ...preset.config,
          showLabels: prev.showLabels,
        }));
      }, 0);
    };
    img.onerror = () => {
      alert(`Failed to load sample image: ${preset.imagePath}\nMake sure the image exists in the public/samples folder.`);
    };
    img.src = preset.imagePath;
  };

  const updateConfig = (key: keyof HexConfig, value: number | boolean) => {
    setHexConfig(prev => {
        const next = { ...prev, [key]: value };
        // Recalculate grid if image is loaded and geometry changes
        if (typeof value === 'number' && imageDims.width > 0 && (key === 'width' || key === 'height')) {
            const dims = calculateGridDimensions(imageDims.width, imageDims.height, next.width, next.height);
            next.gridWidth = dims.gridWidth;
            next.gridHeight = dims.gridHeight;
        }
        return next;
    });
  };

  const handleAutoCalibrate = () => {
      if (!imageRef.current) return;
      
      const detected = detectGrid(imageRef.current);
      if (detected.width && detected.height) {
          setHexConfig(prev => {
              const next = { ...prev, ...detected };
              // Recalc grid dimensions
              const dims = calculateGridDimensions(imageDims.width, imageDims.height, next.width!, next.height!);
              return { ...next, gridWidth: dims.gridWidth, gridHeight: dims.gridHeight };
          });
      } else {
          alert("Could not detect grid pattern. Ensure image has visible hex borders.");
      }
  };

  const handleMapClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !imageSrc || status === ProcessingStatus.PROCESSING) return;

    // We need to account for the zoom scaling in our coordinate calculation
    // e.nativeEvent.offsetX/Y gives position relative to the target element (the scaled div)
    // dividing by zoom gives us the coordinates in the original image space
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const xInScaled = (e.clientX - rect.left); 
    const yInScaled = (e.clientY - rect.top);

    // Convert to unscaled image space
    const x = xInScaled / zoom;
    const y = yInScaled / zoom;

    // Brute force closest hex
    let closestD = Infinity;
    let closestTile = { q: 0, r: 0, x: 0, y: 0 };
    
    const coords = generateGridCoords(hexConfig);
    for (const c of coords) {
        const d = (c.x - x) ** 2 + (c.y - y) ** 2;
        if (d < closestD) {
            closestD = d;
            closestTile = c;
        }
    }

    if (closestD < (hexConfig.width * hexConfig.width)) {
        const id = `${closestTile.q},${closestTile.r}`;
        setHighlightedTile({ q: closestTile.q, r: closestTile.r });
        
        const base64 = cropTile(imageRef.current, closestTile, hexConfig.width, hexConfig.height);
        
        if (base64) {
            let existingLabels = getExampleById(id);
            
            if (existingLabels) {
                 const sanitize = (val: string | undefined) => (!val || val === 'undefined') ? 'None' : val;
                 existingLabels = {
                     terrain: sanitize(existingLabels.terrain),
                     unit: sanitize(existingLabels.unit),
                     tileOwner: sanitize(existingLabels.tileOwner),
                     unitOwner: sanitize(existingLabels.unitOwner),
                     infrastructure: sanitize(existingLabels.infrastructure)
                 };
            }
            
            let predictedLabels;
            try {
                const img = await loadImageFromBase64(base64);
                const pred = await predictTile(img);
                predictedLabels = pred.labels;
            } catch (e) {
                console.warn("Prediction failed in inspector", e);
            }

            setInspectorTile({ 
                ...closestTile, 
                base64,
                existingLabels,
                predictedLabels
            });
        }
    }
  };
  
  const handleScrollSync = (source: 'original' | 'recon') => {
      if (viewMode !== 'split') return;
      if (isSyncingScroll.current) return;

      isSyncingScroll.current = true;
      const srcEl = source === 'original' ? mapContainerRef.current : reconContainerRef.current;
      const tgtEl = source === 'original' ? reconContainerRef.current : mapContainerRef.current;

      if (srcEl && tgtEl) {
          tgtEl.scrollTop = srcEl.scrollTop;
          tgtEl.scrollLeft = srcEl.scrollLeft;
      }
      
      // Debounce the lock release slightly
      setTimeout(() => {
          isSyncingScroll.current = false;
      }, 50);
  };

  const handleResultClick = useCallback((q: number, r: number) => {
      setHighlightedTile({ q, r });
      const { x, y } = getHexCenter(q, r, hexConfig);
      
      const scrollTo = (container: HTMLDivElement | null) => {
        if (container) {
            container.scrollTo({
                left: (x * zoom) - container.clientWidth / 2,
                top: (y * zoom) - container.clientHeight / 2,
                behavior: 'smooth'
            });
        }
      };

      scrollTo(mapContainerRef.current);
      if (viewMode === 'split' || viewMode === 'reconstructed') {
          scrollTo(reconContainerRef.current);
      }
      
      // Also scroll sidebar item into view
      setTimeout(() => {
          const el = document.getElementById(`result-item-${q}-${r}`);
          el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }, 50);
  }, [hexConfig, zoom, viewMode]);

  // Keyboard Navigation for Results
  useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
          // Ignore if user is typing in an input
          if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) return;
          
          if (displayResults.length === 0) return;

          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.preventDefault();
              
              let newIndex = 0;
              
              if (highlightedTile) {
                  const currIndex = displayResults.findIndex(r => r.q === highlightedTile.q && r.r === highlightedTile.r);
                  if (currIndex !== -1) {
                      if (e.key === 'ArrowDown') newIndex = Math.min(displayResults.length - 1, currIndex + 1);
                      if (e.key === 'ArrowUp') newIndex = Math.max(0, currIndex - 1);
                  }
              }
              
              const target = displayResults[newIndex];
              if (target) {
                  handleResultClick(target.q, target.r);
              }
          }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
  }, [displayResults, highlightedTile, handleResultClick]);

  const handleSaveTile = async (props: TileProperties) => {
      if (!inspectorTile) return;

      // Use image hash as ID instead of q/r coordinates
      // This allows training on multiple maps without coordinate conflicts
      const id = hashBase64Image(inspectorTile.base64);
      setStatus(ProcessingStatus.TRAINING);
      await trainTile(inspectorTile.base64, props, id);
      setLabelStats(getLabelCounts()); // Update Sidebar
      setStatus(ProcessingStatus.IDLE);
      setInspectorTile(null);
  };

  const handleRenameLabel = async (category: keyof TileProperties, oldLabel: string, newLabel: string) => {
      setStatus(ProcessingStatus.TRAINING);
      await renameLabelInDataset(category, oldLabel, newLabel);
      setLabelStats(getLabelCounts()); // Update Sidebar
      setStatus(ProcessingStatus.IDLE);
  };

  const handleDeleteLabel = async (category: keyof TileProperties, label: string) => {
      setStatus(ProcessingStatus.TRAINING);
      await deleteLabelFromDataset(category, label);
      setLabelStats(getLabelCounts()); // Update Sidebar
      setStatus(ProcessingStatus.IDLE);
  };
  
  const handleSelectLabel = (category: keyof TileProperties, label: string) => {
      if (activeLabelFilter?.category === category && activeLabelFilter?.label === label) {
          setActiveLabelFilter(null); // Toggle off
      } else {
          setActiveLabelFilter({ category, label });
      }
  };

  // Training Data Viewer handlers
  const openTrainingViewer = () => {
      setTrainingExamples(getAllExamples());
      setShowTrainingViewer(true);
  };

  const handleEditExample = (example: TrainingExample) => {
      // Open in inspector for editing
      setInspectorTile({
          q: 0,
          r: 0,
          x: 0,
          y: 0,
          base64: example.imageBase64,
          existingLabels: example.labels,
          predictedLabels: undefined,
      });
      setShowTrainingViewer(false);
  };

  const handleDeleteExample = async (id: string) => {
      setStatus(ProcessingStatus.TRAINING);
      await deleteExampleById(id);
      setTrainingExamples(getAllExamples());
      setLabelStats(getLabelCounts());
      setStatus(ProcessingStatus.IDLE);
  };

  const handleResetKB = () => {
      if (confirm('Reset Knowledge Base? This will clear all training examples.')) {
          resetDataset();
          setLabelStats(getLabelCounts());
          setLoadedKBName(null);
          setResults([]);
      }
  };

  const handleExport = () => {
      const kb = getDataset(activeLabelsForInspector);
      const blob = new Blob([JSON.stringify(kb)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `weemap-kb-${Date.now()}.json`;
      a.click();
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
          const filename = file.name;
          const reader = new FileReader();
          reader.onload = async (ev) => {
              try {
                  setStatus(ProcessingStatus.TRAINING);
                  const kb = JSON.parse(ev.target?.result as string) as KnowledgeBase;
                  if (kb.examples) {
                      await importDataset({ ...kb, labels: activeLabelsForInspector });
                      setLabelStats(getLabelCounts());
                      setLoadedKBName(filename);
                      alert(`Imported ${kb.examples.length} examples from "${filename}".`);
                  }
                  setStatus(ProcessingStatus.IDLE);
              } catch (err) {
                  console.error(err);
                  alert("Failed to import knowledge base.");
                  setStatus(ProcessingStatus.IDLE);
              }
          };
          reader.readAsText(file);
      }
  };

  const runAnalysis = async () => {
    // If viewMode is Reconstructed, imageRef might be null. 
    // We can't analyze if we can't see the image.
    if (!imageRef.current || !isModelReady) {
        if (!imageRef.current && imageSrc) {
            alert("Please switch to 'Original' or 'Split' view to run analysis.");
        }
        return;
    }
    
    setStatus(ProcessingStatus.PROCESSING);
    setResults([]);
    setProgress(0);
    abortControllerRef.current = new AbortController();

    const coords = generateGridCoords(hexConfig);
    const totalTiles = coords.length;
    let processedCount = 0;
    const batchSize = 10;

    const process = async () => {
        let batch: typeof coords = [];
        
        for (let i = 0; i < coords.length; i++) {
            if (abortControllerRef.current?.signal.aborted) break;
            
            // Critical check: if user switched view mode during processing, 
            // imageRef.current might be null.
            if (!imageRef.current) {
                console.warn("Analysis stopped: Source image not available (View changed?)");
                break;
            }
            
            batch.push(coords[i]);
            
            if (batch.length >= batchSize || i === coords.length - 1) {
                 const batchResults: AnalysisResult[] = [];
                 
                 for (const c of batch) {
                    // Re-check inside inner loop for safety
                    if (!imageRef.current) break;

                    const center = { x: c.x, y: c.y };
                    const base64 = cropTile(imageRef.current, center, hexConfig.width, hexConfig.height);
                    if (base64) {
                        const img = await loadImageFromBase64(base64);
                        const pred = await predictTile(img);
                        batchResults.push({
                            q: c.q, r: c.r, x: c.x, y: c.y,
                            ...pred.labels,
                            confidence: pred.confidence
                        });
                    }
                 }
                 
                 setResults(prev => [...prev, ...batchResults]);
                 processedCount += batch.length;
                 setProgress((processedCount / totalTiles) * 100);
                 batch = [];
                 await new Promise(r => setTimeout(r, 0));
            }
        }
        
        if (!abortControllerRef.current?.signal.aborted) {
            setStatus(ProcessingStatus.COMPLETE);
        } else {
            setStatus(ProcessingStatus.IDLE);
        }
    };
    
    process();
  };

  // --- Render ---

  const hlWidth = hexConfig.width + 16;
  const hlHeight = hexConfig.height + 16;
  const filterPathData = getHexPathData(hexConfig.width, hexConfig.height);

  return (
    <div className="h-screen w-screen bg-gray-900 text-gray-100 flex flex-col font-sans overflow-hidden">
      
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-3 shadow-md shrink-0 z-50">
        <div className="flex justify-between items-center px-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg">H</div>
            <h1 className="text-xl font-bold tracking-tight">WeeMap Scanner <span className="text-gray-500 font-normal text-sm ml-2">Local Learning</span></h1>
          </div>
          <div className="flex gap-4 items-center">
             {/* View Mode Toggle */}
             <div className="flex bg-gray-700 rounded-lg p-1 gap-1">
                 <button 
                    onClick={() => setViewMode('original')}
                    className={`px-3 py-1 text-xs rounded-md transition-colors flex items-center gap-1 ${viewMode === 'original' ? 'bg-gray-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
                 >
                     <EyeIcon className="w-3 h-3"/> Original
                 </button>
                 <button 
                    onClick={() => setViewMode('split')}
                    className={`px-3 py-1 text-xs rounded-md transition-colors flex items-center gap-1 ${viewMode === 'split' ? 'bg-gray-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
                 >
                     <Square2StackIcon className="w-3 h-3"/> Split
                 </button>
                 <button 
                    onClick={() => setViewMode('reconstructed')}
                    className={`px-3 py-1 text-xs rounded-md transition-colors flex items-center gap-1 ${viewMode === 'reconstructed' ? 'bg-gray-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
                 >
                     <SparklesIcon className="w-3 h-3"/> Recon
                 </button>
             </div>

             <div className="w-px h-6 bg-gray-700 mx-2"></div>

             <div className="text-xs text-gray-400 mr-4 font-mono">
                 {status === ProcessingStatus.TRAINING ? "Training..." : status === ProcessingStatus.PROCESSING ? `Processing ${Math.round(progress)}%` : isModelReady ? "Ready" : "Loading..."}
             </div>
             {status === ProcessingStatus.PROCESSING ? (
                 <button 
                  onClick={() => abortControllerRef.current?.abort()}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 rounded-md font-medium text-sm transition-colors"
                 >
                   <StopIcon className="w-5 h-5" /> Stop
                 </button>
             ) : (
                <button 
                  disabled={!imageSrc || !isModelReady}
                  onClick={runAnalysis}
                  className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-colors shadow-lg
                    ${(!imageSrc || !isModelReady) ? 'bg-gray-700 text-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-600/20'}`}
                >
                  <PlayIcon className="w-5 h-5" /> Analyze Map
                </button>
             )}
          </div>
        </div>
      </header>

      {/* Main Body - Border Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* West Panel: Knowledge Base */}
        <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col shrink-0 z-20 shadow-xl">
            <KnowledgeBasePanel
                labelStats={labelStats}
                activeFilter={activeLabelFilter}
                onImageUpload={handleImageUpload}
                onImport={handleImport}
                onExport={handleExport}
                onDeleteLabel={handleDeleteLabel}
                onRenameLabel={handleRenameLabel}
                onSelectLabel={handleSelectLabel}
                onLoadSample={loadSamplePreset}
                onViewTrainingData={openTrainingViewer}
                onReset={handleResetKB}
                exampleCount={getAllExamples().length}
                loadedKBName={loadedKBName}
            />
        </aside>

        {/* Center Panel: Map View (Split or Single) */}
        <main className="flex-1 bg-gray-900 relative flex flex-col min-w-0 overflow-hidden">
           {/* Top Info Bar */}
           <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/90 backdrop-blur border-b border-gray-700 p-2 flex justify-between items-center px-4 shadow-sm">
               <span className="text-xs font-mono text-gray-400">
                   {imageSrc ? `${imageDims.width}x${imageDims.height}px` : "No Image"}
               </span>
               <div className="flex items-center gap-2 text-xs text-gray-400">
                   <span>Scale: {Math.round(zoom * 100)}%</span>
               </div>
               <div className="text-xs text-gray-400">
                   {activeLabelFilter && <span className="text-blue-400 font-bold mr-2">Filter: {activeLabelFilter.label}</span>}
                   {displayResults.length > 0 ? `${displayResults.length} tiles analyzed` : "Click tiles to train"}
               </div>
           </div>
           
           <div className="flex-1 flex overflow-hidden relative mt-[41px]">
               
               {/* ORIGINAL MAP VIEW */}
               {(viewMode === 'original' || viewMode === 'split') && (
                   <div 
                     ref={mapContainerRef}
                     onScroll={() => handleScrollSync('original')}
                     className={`flex-1 overflow-auto relative cursor-crosshair bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiMzMzMiLz48L3N2Zz4=')] bg-repeat ${viewMode === 'split' ? 'border-r border-gray-700' : ''}`}
                     onClick={handleMapClick}
                   >
                      {imageSrc ? (
                         <div 
                            className="relative inline-block origin-top-left transition-transform duration-150 ease-out" 
                            style={{ 
                                width: imageDims.width, 
                                height: imageDims.height,
                                transform: `scale(${zoom})`,
                                transformOrigin: '0 0'
                            }}
                        >
                             <img
                               ref={imageRef}
                               src={imageSrc}
                               alt="Map"
                               width={imageDims.width}
                               height={imageDims.height}
                               className={`block max-w-none transition-opacity duration-300 ${activeLabelFilter ? 'opacity-50' : 'opacity-100'}`}
                               style={{ imageRendering: 'pixelated' }}
                             />
                             
                             <GridOverlay
                                key={`${imageDims.width}-${imageDims.height}-${hexConfig.originX}-${hexConfig.originY}-${hexConfig.width}-${hexConfig.height}`}
                                config={hexConfig}
                                imageWidth={imageDims.width}
                                imageHeight={imageDims.height}
                             />
                             
                             {/* Results Overlay (Dots & Labels) */}
                             {results.map((r, i) => {
                                 if (r.terrain === 'Unknown' || r.terrain === 'Empty') return null;
                                 const center = getHexCenter(r.q, r.r, hexConfig);
                                 
                                 // Check Filter
                                 let isMatch = false;
                                 if (activeLabelFilter) {
                                     const val = (r as any)[activeLabelFilter.category];
                                     isMatch = val === activeLabelFilter.label;
                                 }

                                 return (
                                     <React.Fragment key={`${r.q},${r.r}`}>
                                        <div 
                                            className={`absolute pointer-events-none transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center ${activeLabelFilter && !isMatch ? 'opacity-20' : 'opacity-100'}`}
                                            style={{ left: center.x, top: center.y }}
                                        >
                                             {r.unit !== 'None' && r.unit !== 'undefined' && <div className={`w-2 h-2 rounded-full mb-0.5 shadow-sm border border-white ${r.unitOwner === 'Red' ? 'bg-red-500' : r.unitOwner === 'Blue' ? 'bg-blue-500' : 'bg-gray-400'}`}></div>}
                                             {r.unit !== 'None' && r.unit !== 'undefined' && <span className="text-[8px] leading-none bg-black/60 px-0.5 rounded text-white backdrop-blur-sm whitespace-nowrap">{r.unit}</span>}
                                             {r.infrastructure !== 'None' && r.infrastructure !== 'undefined' && <span className="text-[6px] leading-none text-yellow-300 mt-0.5">{r.infrastructure}</span>}
                                         </div>
                                         
                                         {/* Filter Match Highlight */}
                                         {isMatch && (
                                             <div 
                                                className="absolute pointer-events-none transform -translate-x-1/2 -translate-y-1/2 z-10"
                                                style={{ left: center.x, top: center.y, width: hlWidth, height: hlHeight }}
                                             >
                                                <svg width={hlWidth} height={hlHeight} style={{ overflow: 'visible' }}>
                                                    <g transform={`translate(${hlWidth/2}, ${hlHeight/2})`}>
                                                        <path 
                                                            d={filterPathData} 
                                                            fill="rgba(236, 72, 153, 0.2)" 
                                                            stroke="#ec4899" 
                                                            strokeWidth="2" 
                                                        />
                                                    </g>
                                                </svg>
                                             </div>
                                         )}
                                     </React.Fragment>
                                 )
                             })}

                             {/* Highlight Overlay (Current Selection) */}
                             {highlightedTile && (
                                <div 
                                    className="absolute pointer-events-none z-50 transform -translate-x-1/2 -translate-y-1/2"
                                    style={{ 
                                        left: getHexCenter(highlightedTile.q, highlightedTile.r, hexConfig).x, 
                                        top: getHexCenter(highlightedTile.q, highlightedTile.r, hexConfig).y,
                                        width: hlWidth,
                                        height: hlHeight
                                    }}
                                >
                                    <svg 
                                        width={hlWidth} 
                                        height={hlHeight} 
                                        style={{ overflow: 'visible' }}
                                    >
                                        <g transform={`translate(${hlWidth/2}, ${hlHeight/2})`}>
                                            <path 
                                                d={getHexPathData(hexConfig.width, hexConfig.height)} 
                                                fill="none" 
                                                stroke="#fbbf24" 
                                                strokeWidth="3" 
                                                className="animate-pulse drop-shadow-[0_0_8px_rgba(251,191,36,0.8)]"
                                            />
                                        </g>
                                    </svg>
                                </div>
                             )}
                         </div>
                      ) : (
                        <div className="h-full flex items-center justify-center text-gray-600 flex-col gap-4">
                            <CloudArrowUpIcon className="w-16 h-16 text-gray-700" />
                            <p>Upload a screenshot from the left panel to start</p>
                        </div>
                      )}
                   </div>
               )}

               {/* RECONSTRUCTED MAP VIEW */}
               {(viewMode === 'split' || viewMode === 'reconstructed') && (
                   <div 
                        ref={reconContainerRef}
                        onScroll={() => handleScrollSync('recon')}
                        className="flex-1 overflow-auto bg-gray-950 relative"
                   >
                       {imageSrc ? (
                           <div style={{ width: imageDims.width * zoom, height: imageDims.height * zoom }}>
                               <ReconstructedMap 
                                   results={results}
                                   config={hexConfig}
                                   width={imageDims.width}
                                   height={imageDims.height}
                                   zoom={zoom}
                               />
                               {/* Share Highlight on Recon Map */}
                               {highlightedTile && (
                                    <div 
                                        className="absolute pointer-events-none z-50 transform -translate-x-1/2 -translate-y-1/2"
                                        style={{ 
                                            left: getHexCenter(highlightedTile.q, highlightedTile.r, hexConfig).x * zoom, 
                                            top: getHexCenter(highlightedTile.q, highlightedTile.r, hexConfig).y * zoom,
                                            width: hlWidth * zoom,
                                            height: hlHeight * zoom
                                        }}
                                    >
                                        <svg 
                                            width={hlWidth * zoom} 
                                            height={hlHeight * zoom} 
                                            style={{ overflow: 'visible' }}
                                            viewBox={`0 0 ${hlWidth} ${hlHeight}`}
                                        >
                                            <g transform={`translate(${hlWidth/2}, ${hlHeight/2})`}>
                                                <path 
                                                    d={getHexPathData(hexConfig.width, hexConfig.height)} 
                                                    fill="none" 
                                                    stroke="#fbbf24" 
                                                    strokeWidth="3" 
                                                    className="animate-pulse"
                                                />
                                            </g>
                                        </svg>
                                    </div>
                                )}
                           </div>
                       ) : (
                           <div className="h-full flex items-center justify-center text-gray-700">
                               Waiting for Map...
                           </div>
                       )}
                   </div>
               )}
               
           </div>
        </main>

        {/* East Panel: Calibration & Results */}
        <aside className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col shrink-0 z-20 shadow-xl">
           
           {/* Grid Calibration Section */}
           <div className="p-4 border-b border-gray-700 bg-gray-800">
               <div className="flex justify-between items-center mb-4">
                   <h2 className="text-xs font-bold uppercase text-gray-400 flex items-center gap-2">
                       <AdjustmentsHorizontalIcon className="w-4 h-4" /> Grid Calibration
                   </h2>
                   <div className="flex gap-2">
                       <button onClick={handleAutoCalibrate} className="text-[10px] bg-blue-900 text-blue-300 px-2 py-1 rounded hover:bg-blue-800 border border-blue-700">
                           Auto-Detect
                       </button>
                   </div>
               </div>
               
               <div className="space-y-4">
                   <div className="grid grid-cols-2 gap-3">
                       <DebouncedNumberInput label="Origin X" value={hexConfig.originX} onChange={v => updateConfig('originX', v)} />
                       <DebouncedNumberInput label="Origin Y" value={hexConfig.originY} onChange={v => updateConfig('originY', v)} />
                       <DebouncedNumberInput label="Hex Width" value={hexConfig.width} onChange={v => updateConfig('width', v)} />
                       <DebouncedNumberInput label="Hex Height" value={hexConfig.height} onChange={v => updateConfig('height', v)} />
                   </div>

                   <div className="pt-2 border-t border-gray-700 space-y-3">
                        {/* Zoom Control */}
                        <div>
                             <div className="flex justify-between text-xs text-gray-500 mb-1">
                                 <span>View Zoom</span>
                                 <span>{Math.round(zoom * 100)}%</span>
                             </div>
                             <input 
                                 type="range" 
                                 min="0.5" 
                                 max="3.0" 
                                 step="0.1" 
                                 value={zoom} 
                                 onChange={(e) => setZoom(parseFloat(e.target.value))}
                                 className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                             />
                        </div>

                        {/* Labels Toggle */}
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-400">Show Q,R Labels</span>
                            <input 
                                type="checkbox" 
                                checked={hexConfig.showLabels} 
                                onChange={(e) => updateConfig('showLabels', e.target.checked)}
                                className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-gray-800"
                            />
                        </div>
                   </div>
               </div>
           </div>

           {/* Analysis Results List */}
           <div className="flex-1 flex flex-col min-h-0">
               <div className="p-3 bg-gray-900 border-b border-gray-700">
                   <h2 className="text-xs font-bold uppercase text-gray-400 flex items-center gap-2">
                       <MagnifyingGlassPlusIcon className="w-4 h-4" /> Analyzed Tiles
                   </h2>
               </div>
               
               <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                   {displayResults.length === 0 ? (
                       <div className="text-center text-gray-600 text-xs mt-10 p-4">
                           {status === ProcessingStatus.PROCESSING ? "Scanning..." : "No results yet. Click 'Analyze Map' to start."}
                       </div>
                   ) : (
                       <div className="space-y-1">
                           {displayResults.map((res) => {
                               const isActive = highlightedTile?.q === res.q && highlightedTile?.r === res.r;
                               return (
                                   <div 
                                       id={`result-item-${res.q}-${res.r}`}
                                       key={`${res.q}-${res.r}`}
                                       onClick={() => handleResultClick(res.q, res.r)}
                                       className={`p-2 rounded cursor-pointer border flex flex-col gap-1 transition-colors
                                            ${isActive ? 'bg-blue-900/40 border-blue-500 ring-1 ring-blue-500/50' : 'bg-gray-800 border-gray-700 hover:bg-gray-700'}`}
                                   >
                                       <div className="flex justify-between items-center">
                                           <span className="font-mono text-xs text-blue-400 bg-blue-900/30 px-1.5 rounded">
                                               {res.q}, {res.r}
                                           </span>
                                           <div className="flex gap-1">
                                               {res.confidence < 0.7 && <span className="text-[10px] text-yellow-500" title="Low Confidence">⚠️</span>}
                                               <span className="text-[10px] text-gray-500">{(res.confidence * 100).toFixed(0)}%</span>
                                           </div>
                                       </div>
                                       
                                       <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
                                            <div className="text-gray-300"><span className="text-gray-500 mr-1">T:</span>{res.terrain}</div>
                                            <div className="text-gray-300"><span className="text-gray-500 mr-1">U:</span>{res.unit === 'None' ? '-' : res.unit}</div>
                                            <div className="text-gray-400"><span className="text-gray-600 mr-1">TO:</span>{res.tileOwner === 'None' ? '-' : res.tileOwner}</div>
                                            <div className="text-gray-400"><span className="text-gray-600 mr-1">UO:</span>{res.unitOwner === 'None' ? '-' : res.unitOwner}</div>
                                            {res.infrastructure !== 'None' && <div className="col-span-2 text-yellow-500/80"><span className="text-gray-600 mr-1">I:</span>{res.infrastructure}</div>}
                                       </div>
                                   </div>
                               );
                           })}
                       </div>
                   )}
               </div>
           </div>

        </aside>

        {/* Floating Modal for Tile Inspector */}
        {inspectorTile && (
            <TileInspector
                tile={inspectorTile}
                activeLabels={activeLabelsForInspector}
                onSave={handleSaveTile}
                onClose={() => setInspectorTile(null)}
            />
        )}

        {/* Training Data Viewer Modal */}
        {showTrainingViewer && (
            <TrainingDataViewer
                examples={trainingExamples}
                labelStats={labelStats}
                onClose={() => setShowTrainingViewer(false)}
                onEdit={handleEditExample}
                onDelete={handleDeleteExample}
            />
        )}

      </div>
    </div>
  );
};

export default App;
