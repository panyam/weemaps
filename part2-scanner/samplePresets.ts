/**
 * Sample map presets with pre-configured hex grid settings.
 * Add your sample images to /public/samples/ and update this file.
 *
 * To add a new sample:
 * 1. Take a screenshot of the game map
 * 2. Save it to /public/samples/
 * 3. Add an entry here with calibrated origin/width/height values
 * 4. Use the app's calibration tools to find the right values
 */

/**
 * Default Knowledge Base to load on startup.
 * Set to null to start with empty KB, or provide a path like '/samples/default-kb.json'
 */
const PATH_PREFIX = "" // "/demos/weemap-scanner"
export const defaultKBPath: string | null = PATH_PREFIX + '/samples/weewar-model.json';

export interface SamplePreset {
  id: string;
  name: string;
  description: string;
  imagePath: string;
  config: {
    originX: number;
    originY: number;
    width: number;
    height: number;
    gridWidth: number;
    gridHeight: number;
  };
}

export const samplePresets: SamplePreset[] = [
  // WeeWar / TinyAttack style games
  {
    id: 'weewar-1',
    name: 'WeeWar - The Avocado Jungle of Death',
    description: 'A standard WeeWar battlefield',
    imagePath: PATH_PREFIX + '/samples/weewar/19771.png',
    config: {
      originX: 16,
      originY: 16,
      width: 32,
      height: 34.75,
      gridWidth: 20,
      gridHeight: 15,
    },
  },
  {
    id: 'weewar-2',
    name: 'WeeWar - Classic Map',
    description: 'Higher Straits',
    imagePath: PATH_PREFIX + '/samples/weewar/20874.png',
    config: {
      originX: 16,
      originY: 17,
      width: 32,
      height: 34.65,
      gridWidth: 20,
      gridHeight: 15,
    },
  },
  {
    id: 'weewar-3',
    name: 'WeeWar - Four Players',
    description: 'Higher Straits',
    imagePath: PATH_PREFIX + '/samples/weewar/preset3.png',
    config: {
      originX: 0,
      originY: 32,
      width: 64,
      height: 69.5,
      gridWidth: 64,
      gridHeight: 69.5
    },
  }
];
