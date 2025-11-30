import { HexConfig } from '../types';

/**
 * A heuristic-based solver to detect hex grid dimensions from a screenshot.
 * It uses Edge Detection (Sobel-ish) followed by Autocorrelation to find repeated signals.
 */
export const detectGrid = (img: HTMLImageElement): Partial<HexConfig> => {
    // Defensive check
    if (!img || img.naturalWidth === 0 || img.naturalHeight === 0) return {};

    // 1. Downscale for performance (work on a ~600px wide canvas)
    const MAX_WIDTH = 600;
    const scale = Math.min(1, MAX_WIDTH / img.width);
    
    const w = Math.floor(img.width * scale);
    const h = Math.floor(img.height * scale);
    
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (!ctx) return {};

    ctx.drawImage(img, 0, 0, w, h);
    
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    // 2. Compute Row and Column Energy (Edge Density)
    // We are looking for "high frequency" areas (edges of hexes)
    const rowEnergy = new Float32Array(h);
    const colEnergy = new Float32Array(w);

    // Simple derivative filter [ -1, 0, 1 ] to find edges
    for (let y = 0; y < h; y++) {
        for (let x = 1; x < w - 1; x++) {
            const i = (y * w + x) * 4;
            const i_prev = (y * w + (x - 1)) * 4;
            const i_next = (y * w + (x + 1)) * 4;

            // Luminance
            const l = (data[i] + data[i+1] + data[i+2]) / 3;
            const l_prev = (data[i_prev] + data[i_prev+1] + data[i_prev+2]) / 3;
            const l_next = (data[i_next] + data[i_next+1] + data[i_next+2]) / 3;

            const edgeVal = Math.abs(l_next - l_prev);
            
            rowEnergy[y] += edgeVal;
            colEnergy[x] += edgeVal;
        }
    }

    // 3. Autocorrelation to find Periodicity (Grid Stride)
    const findPeriod = (signal: Float32Array, minP: number, maxP: number) => {
        let bestPeriod = 0;
        let bestScore = -Infinity;

        // Normalize signal average to 0 to avoid DC bias
        let sum = 0;
        for(let i=0; i<signal.length; i++) sum += signal[i];
        const avg = sum / signal.length;
        const normSignal = new Float32Array(signal.length);
        for(let i=0; i<signal.length; i++) normSignal[i] = signal[i] - avg;

        for (let period = minP; period <= maxP; period++) {
            let score = 0;
            let count = 0;
            // Check correlation at k*period
            for (let k = 1; k < 4; k++) {
                const shift = period * k;
                for (let i = 0; i < normSignal.length - shift; i++) {
                    score += normSignal[i] * normSignal[i + shift];
                    count++;
                }
            }
            if (count > 0) score /= count;

            if (score > bestScore) {
                bestScore = score;
                bestPeriod = period;
            }
        }
        return bestPeriod;
    };

    // Limits for hex size (scaled)
    const minDim = 10; 
    const maxDim = w / 5;

    const periodX = findPeriod(colEnergy, minDim, maxDim);
    const periodY = findPeriod(rowEnergy, minDim, maxDim);

    // 4. Find Phase (Origin) - Where does the pattern start?
    // We look for the index that maximizes sum(signal[offset + k*period])
    const findPhase = (signal: Float32Array, period: number) => {
        let bestPhase = 0;
        let maxEnergy = -Infinity;

        for (let phase = 0; phase < period; phase++) {
            let energy = 0;
            for (let i = phase; i < signal.length; i += period) {
                energy += signal[i];
            }
            if (energy > maxEnergy) {
                maxEnergy = energy;
                bestPhase = phase;
            }
        }
        return bestPhase;
    };

    const phaseX = findPhase(colEnergy, periodX);
    const phaseY = findPhase(rowEnergy, periodY);

    // 5. Scale back up to original image dimensions
    // For pointy top hexes:
    // periodX usually corresponds to Width (or Width/2 depending on pattern, but usually Width for autocorrelation)
    // periodY usually corresponds to Height * 0.75
    
    // Note: Autocorrelation finds the STRIDE.
    // Horizontal stride for pointy top is Width.
    // Vertical stride for pointy top is 0.75 * Height.

    const detectedWidth = periodX / scale;
    const detectedHeight = (periodY / scale) / 0.75;
    
    const originX = phaseX / scale;
    const originY = phaseY / scale;

    return {
        width: parseFloat(detectedWidth.toFixed(1)),
        height: parseFloat(detectedHeight.toFixed(1)),
        originX: parseFloat(originX.toFixed(1)),
        originY: parseFloat(originY.toFixed(1))
    };
};