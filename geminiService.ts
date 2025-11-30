import { GoogleGenAI, Type, Schema, Part } from "@google/genai";
import { AnalysisResult } from "../types";

/**
 * Few-shot example for teaching Gemini about specific tile types.
 *
 * Note: This approach has limitations compared to the local KNN approach:
 * - Each API call must re-send all examples (expensive at scale)
 * - With 50+ tile/unit types, you can't embed them all
 * - No persistent "memory" between calls
 *
 * The local MobileNet+KNN approach is better for this use case because:
 * - Training is instant and persistent
 * - No per-call cost
 * - Works offline
 * - User can teach it any label they want
 */
export interface FewShotExample {
  imageBase64: string;  // Base64 encoded image (with or without data URL prefix)
  terrain: string;      // e.g., "Grass", "Mountain", "Water"
  unit?: string;        // e.g., "Tank", "Infantry", or undefined for no unit
  owner?: string;       // e.g., "Red", "Blue", "Neutral"
}

const TILE_ANALYSIS_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    tileType: {
      type: Type.STRING,
      description: "The terrain type of the hex tile (e.g., Grass, Water, Mountain, Forest, City, Factory, Base, Shoal, Reef).",
    },
    unit: {
      type: Type.STRING,
      description: "The name of the military unit on the tile, if any (e.g., Infantry, Tank, Helicopter, Mech, Artillery). Return 'None' if empty.",
    },
    player: {
      type: Type.STRING,
      description: "The color or faction of the unit or city (e.g., Red, Blue, Green, Yellow). Return 'Neutral' for uncaptured properties, or 'None' for nature.",
    },
    infrastructure: {
      type: Type.STRING,
      description: "Any road, bridge, or river crossing on the tile. Return 'Road', 'Bridge', or 'None'.",
    },
    confidence: {
      type: Type.NUMBER,
      description: "A number between 0 and 1 indicating confidence in the classification.",
    }
  },
  required: ["tileType", "unit", "player", "infrastructure"],
};

/**
 * Build the few-shot prompt parts from examples.
 * Each example becomes an image + text pair showing the expected classification.
 */
const buildFewShotParts = (examples: FewShotExample[]): Part[] => {
  const parts: Part[] = [];

  if (examples.length === 0) {
    return parts;
  }

  parts.push({
    text: "Here are some labeled examples from this game to help you understand the visual style:\n"
  });

  for (const example of examples) {
    const cleanBase64 = example.imageBase64.split(',')[1] || example.imageBase64;

    // Add the example image
    parts.push({
      inlineData: {
        mimeType: "image/png",
        data: cleanBase64,
      },
    });

    // Add the label for this example
    const unitPart = example.unit ? `, Unit: ${example.unit}` : "";
    const ownerPart = example.owner ? `, Owner: ${example.owner}` : "";
    parts.push({
      text: `This tile is: Terrain: ${example.terrain}${unitPart}${ownerPart}\n`
    });
  }

  parts.push({
    text: "\nNow analyze this new tile using the same labeling conventions:\n"
  });

  return parts;
};

/**
 * Analyze a tile image using Gemini's vision API.
 *
 * @param base64Image - The tile image to analyze
 * @param q - Axial q coordinate
 * @param r - Axial r coordinate
 * @param fewShotExamples - Optional array of labeled examples to guide classification
 */
export const analyzeTileImage = async (
  base64Image: string,
  q: number,
  r: number,
  fewShotExamples: FewShotExample[] = []
): Promise<AnalysisResult> => {
  try {
    const apiKey = import.meta.env.VITE_API_KEY || import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) throw new Error("API Key not found. Set API_KEY or GEMINI_API_KEY environment variable.");

    const ai = new GoogleGenAI({ apiKey });

    // Remove the data URL prefix if present
    const cleanBase64 = base64Image.split(',')[1] || base64Image;

    // Build the prompt parts
    const parts: Part[] = [];

    // Add few-shot examples if provided
    if (fewShotExamples.length > 0) {
      parts.push(...buildFewShotParts(fewShotExamples));
    }

    // Add the target image
    parts.push({
      inlineData: {
        mimeType: "image/png",
        data: cleanBase64,
      },
    });

    // Add the analysis instruction
    const basePrompt = fewShotExamples.length > 0
      ? "Analyze this hex tile using the same labels as the examples above."
      : "Analyze this cropped hex tile from a 2D strategy game. Identify the terrain, any units present, the player color (if applicable), and infrastructure like roads or bridges. Be specific about unit types (e.g., Tank, Infantry).";

    parts.push({ text: basePrompt });

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: { parts },
      config: {
        responseMimeType: "application/json",
        responseSchema: TILE_ANALYSIS_SCHEMA,
        systemInstruction: "You are an expert at analyzing 16-bit and 32-bit pixel art strategy games like WeeWar, Advance Wars, and Civilization. You can distinguish between different army units and terrain types even in low resolution pixel art. When few-shot examples are provided, use the exact same label conventions.",
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response text");

    const data = JSON.parse(text);

    return {
      q,
      r,
      x: 0,
      y: 0,
      terrain: data.tileType || "Unknown",
      unit: data.unit || "None",
      tileOwner: data.player || "None",
      unitOwner: "None",
      infrastructure: data.infrastructure || "None",
      confidence: data.confidence || 0,
    };

  } catch (error) {
    console.error(`Error analyzing tile ${q},${r}:`, error);
    return {
      q,
      r,
      x: 0,
      y: 0,
      terrain: "Error",
      unit: "None",
      tileOwner: "None",
      unitOwner: "None",
      infrastructure: "None",
      confidence: 0,
    };
  }
};

/**
 * Example usage with few-shot examples:
 *
 * const examples: FewShotExample[] = [
 *   { imageBase64: grassTileBase64, terrain: "Grass" },
 *   { imageBase64: waterTileBase64, terrain: "Water" },
 *   { imageBase64: tankOnGrassBase64, terrain: "Grass", unit: "Tank", owner: "Red" },
 *   { imageBase64: infantryBase64, terrain: "Forest", unit: "Infantry", owner: "Blue" },
 * ];
 *
 * const result = await analyzeTileImage(unknownTileBase64, 5, 10, examples);
 */
