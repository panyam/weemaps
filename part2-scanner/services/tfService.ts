
import { TileProperties, TrainingExample, KnowledgeBase, LabelStats } from "../types";
import { loadImageFromBase64 } from "../utils/hexUtils";

// Declare globals for the CDN-loaded libraries
declare global {
  interface Window {
    tf: any;
    mobilenet: any;
    knnClassifier: any;
  }
}

let net: any = null;
// We now have 5 separate classifiers
const classifiers: Record<keyof TileProperties, any> = {
    terrain: null,
    unit: null,
    tileOwner: null,
    unitOwner: null,
    infrastructure: null
};

// Store examples in memory for export
let dataset: TrainingExample[] = [];

export const loadModels = async () => {
  if (net && classifiers.terrain) return;

  console.log('Loading TensorFlow models...');
  
  if (!window.tf || !window.mobilenet || !window.knnClassifier) {
    throw new Error("TensorFlow.js libraries not loaded yet.");
  }

  // Load MobileNet (Feature Extractor)
  net = await window.mobilenet.load();
  
  // Create KNN Classifiers for each category
  classifiers.terrain = window.knnClassifier.create();
  classifiers.unit = window.knnClassifier.create();
  classifiers.tileOwner = window.knnClassifier.create();
  classifiers.unitOwner = window.knnClassifier.create();
  classifiers.infrastructure = window.knnClassifier.create();
  
  console.log('Models loaded.');
};

export const addExample = async (image: HTMLImageElement, properties: TileProperties) => {
  if (!net) await loadModels();
  
  const activation = net.infer(image, true);
  
  // Add example to each classifier with its specific label
  classifiers.terrain.addExample(activation, properties.terrain);
  classifiers.unit.addExample(activation, properties.unit);
  classifiers.tileOwner.addExample(activation, properties.tileOwner);
  classifiers.unitOwner.addExample(activation, properties.unitOwner);
  classifiers.infrastructure.addExample(activation, properties.infrastructure);
  
  activation.dispose();
};

const rebuildClassifiers = async () => {
    if (!net) await loadModels();
    
    // Clear all classifiers
    Object.values(classifiers).forEach(c => c.clearAllClasses());
    
    console.log("Rebuilding classifiers from dataset...", dataset.length);

    // Re-add all examples
    for (const ex of dataset) {
        try {
            const img = await loadImageFromBase64(ex.imageBase64);
            const activation = net.infer(img, true);
            
            classifiers.terrain.addExample(activation, ex.labels.terrain);
            classifiers.unit.addExample(activation, ex.labels.unit);
            classifiers.tileOwner.addExample(activation, ex.labels.tileOwner);
            classifiers.unitOwner.addExample(activation, ex.labels.unitOwner);
            classifiers.infrastructure.addExample(activation, ex.labels.infrastructure);
            
            activation.dispose();
        } catch (e) {
            console.warn("Skipping invalid example during rebuild", e);
        }
    }
    console.log("Rebuild complete.");
};

export const renameLabelInDataset = async (category: keyof TileProperties, oldLabel: string, newLabel: string) => {
    let changed = false;
    dataset.forEach(ex => {
        if (ex.labels[category] === oldLabel) {
            ex.labels[category] = newLabel;
            changed = true;
        }
    });

    if (changed) {
        await rebuildClassifiers();
    }
};

export const deleteLabelFromDataset = async (category: keyof TileProperties, label: string) => {
    // Remove all examples that have this label in this category
    // NOTE: This removes the ENTIRE training example, because an example is a single image.
    // If an image has "Terrain: Grass" and "Unit: Tank", and we delete "Tank", 
    // strictly speaking we should probably just set Unit to "None".
    // However, the prompt says "delete label... remove all test data".
    // Let's implement a safer version: Set to "None" (or default).
    // If the user wants to truly nuke the data, they can manage it differently.
    // Actually, setting to 'None' effectively removes it from the "Label" class in KNN.
    // BUT, if 'None' is a valid label (which it is for Unit), this just re-labels it.
    
    // If the category is mandatory (like Terrain), we might set to 'Unknown' or 'None'.
    
    const defaultVal = category === 'terrain' ? 'Empty' : 'None';
    
    let changed = false;
    // We filter the dataset? No, we modify it.
    // If we want to remove the example entirely if it matches?
    // "Deleting a label should remove all test data for that label."
    
    // Interpretation: If I delete "Tank", all examples that are "Tank" are removed from the training set.
    // This is destructive. Let's do it.
    
    const initialLen = dataset.length;
    dataset = dataset.filter(ex => ex.labels[category] !== label);
    
    if (dataset.length !== initialLen) {
        await rebuildClassifiers();
    }
};

// Helper to re-hydrate from JSON import
export const importDataset = async (kb: KnowledgeBase) => {
    if (!net) await loadModels();

    dataset = []; // Clear current dataset
    console.log(`Importing ${kb.examples.length} examples...`);
    
    // We populate the dataset first
    dataset = [...kb.examples];
    
    // Then rebuild (which clears and re-trains)
    await rebuildClassifiers();
    
    return dataset;
};

export const getDataset = (labels: KnowledgeBase['labels']): KnowledgeBase => {
    return {
        labels,
        examples: dataset
    };
};

export const getExampleById = (id: string): TileProperties | undefined => {
    const ex = dataset.find(e => e.id === id);
    return ex ? ex.labels : undefined;
};

export const getAllExamples = (): TrainingExample[] => {
    return [...dataset];
};

export const resetDataset = () => {
    dataset = [];
    Object.values(classifiers).forEach(c => c?.clearAllClasses());
    console.log("Dataset and classifiers reset.");
};

export const deleteExampleById = async (id: string): Promise<boolean> => {
    const idx = dataset.findIndex(e => e.id === id);
    if (idx === -1) return false;

    dataset.splice(idx, 1);
    await rebuildClassifiers();
    return true;
};

export const updateExampleLabels = async (id: string, newLabels: TileProperties): Promise<boolean> => {
    const ex = dataset.find(e => e.id === id);
    if (!ex) return false;

    ex.labels = newLabels;
    ex.timestamp = Date.now();
    await rebuildClassifiers();
    return true;
};

export const getLabelCounts = (): LabelStats => {
    const stats: LabelStats = {
        terrain: {},
        unit: {},
        tileOwner: {},
        unitOwner: {},
        infrastructure: {}
    };
    
    dataset.forEach(ex => {
        (Object.keys(stats) as Array<keyof TileProperties>).forEach(cat => {
            const label = ex.labels[cat];
            if (label && label !== 'undefined') {
                stats[cat][label] = (stats[cat][label] || 0) + 1;
            }
        });
    });
    
    return stats;
};

// Wrapper to add to dataset array and train
export const trainTile = async (base64: string, properties: TileProperties, id?: string) => {
    const img = await loadImageFromBase64(base64);
    
    // If ID is provided, remove existing example with that ID
    if (id) {
        const idx = dataset.findIndex(e => e.id === id);
        if (idx !== -1) {
            dataset.splice(idx, 1);
        }
    }
    
    dataset.push({
        id,
        imageBase64: base64,
        labels: properties,
        timestamp: Date.now()
    });
    
    // If we removed something, we MUST rebuild to clear the old vector from the KNN
    if (id && dataset.length > 0) {
        await rebuildClassifiers();
    } else {
        await addExample(img, properties);
    }
};

/**
 * Returns current TensorFlow.js memory stats for debugging.
 * Useful for monitoring memory usage in DevTools console:
 *   console.log(window.tf.memory().numTensors)
 */
export const getMemoryStats = () => {
  if (!window.tf) return null;
  return window.tf.memory();
};

export const predictTile = async (image: HTMLImageElement): Promise<{ labels: TileProperties, confidence: number }> => {
  if (!net) throw new Error("Models not loaded");

  const activation = net.infer(image, true);
  
  // Set defaults to None
  const result: TileProperties = {
      terrain: "None",
      unit: "None",
      tileOwner: "None",
      unitOwner: "None",
      infrastructure: "None"
  };

  let totalConf = 0;
  let count = 0;

  // Helper to predict single classifier
  const predictCat = async (cat: keyof TileProperties) => {
      const clf = classifiers[cat];
      if (clf.getNumClasses() > 0) {
          try {
              const res = await clf.predictClass(activation);
              // Handle "undefined" string or undefined value
              if (res.label && res.label !== 'undefined') {
                 result[cat] = res.label;
              } else {
                 result[cat] = 'None';
              }
              
              // If confidence is undefined (1 class only), assume 1
              const conf = res.confidences[res.label] ?? 1;
              totalConf += conf;
              count++;
          } catch (e) { 
              // console.warn(e); 
          }
      }
  };

  await Promise.all([
      predictCat('terrain'),
      predictCat('unit'),
      predictCat('tileOwner'),
      predictCat('unitOwner'),
      predictCat('infrastructure')
  ]);

  activation.dispose();

  return {
    labels: result,
    confidence: count > 0 ? totalConf / count : 0
  };
};
