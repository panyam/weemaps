
export interface HexConfig {
  originX: number;
  originY: number;
  width: number;  // Full width of the hex
  height: number; // Full height of the hex
  gridWidth: number; // Number of columns (in offset coords)
  gridHeight: number; // Number of rows (in offset coords)
  showLabels: boolean; // Toggle Q/R visibility
}

export interface TileCoordinates {
  q: number; // Axial Q
  r: number; // Axial R
  x: number; // Center pixel X
  y: number; // Center pixel Y
}

export interface TileProperties {
  terrain: string;
  unit: string;
  tileOwner: string;
  unitOwner: string;
  infrastructure: string;
}

export interface AnalysisResult extends TileCoordinates, TileProperties {
  confidence: number; // Average confidence
}

export interface KnowledgeBase {
  labels: {
    terrain: string[];
    unit: string[];
    tileOwner: string[];
    unitOwner: string[];
    infrastructure: string[];
  };
  examples: TrainingExample[];
}

export interface LabelStats {
    terrain: Record<string, number>;
    unit: Record<string, number>;
    tileOwner: Record<string, number>;
    unitOwner: Record<string, number>;
    infrastructure: Record<string, number>;
}

export interface TrainingExample {
  id?: string; // Unique identifier (e.g., "q,r")
  imageBase64: string;
  labels: TileProperties;
  timestamp: number;
}

export enum ProcessingStatus {
  IDLE = 'IDLE',
  LOADING_MODEL = 'LOADING_MODEL',
  TRAINING = 'TRAINING',
  PROCESSING = 'PROCESSING',
  COMPLETE = 'COMPLETE',
  ERROR = 'ERROR'
}

export interface InspectorData extends TileCoordinates {
    base64: string;
    existingLabels?: TileProperties;
    predictedLabels?: TileProperties;
}
