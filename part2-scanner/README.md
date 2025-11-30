# WeeMap Scanner AI (Local Learning Edition)

**WeeMap Scanner AI** is a browser-based tool designed to reverse-engineer game states from screenshots of hexagonal strategy games (like Advance Wars, Wesnoth, or Civ). 

It uses **Computer Vision** and **Machine Learning** entirely within your browser (no data is sent to a server) to identify terrain, units, and player ownership.

---

## 🧠 How the AI Works (For ML Beginners)

This application uses a technique called **Transfer Learning** combined with a **K-Nearest Neighbors (KNN)** classifier. Here is a breakdown of the concepts:

### 1. The "Eye": MobileNet (Feature Extraction)
Training an AI from scratch to recognize "tanks" or "grass" requires thousands of images and powerful servers. Instead, we use a pre-trained model called **MobileNet**. 

*   **What it is:** MobileNet is a deep neural network already trained on millions of images (ImageNet) to recognize general objects (dogs, cars, shapes, textures).
*   **How we use it:** We "chop off" the last part of MobileNet that makes the final decision (e.g., "This is a cat"). Instead, we use the internal output: a list of numbers called an **Embedding** or **Feature Vector**.
*   **The Concept:** When MobileNet looks at a "Forest" tile, it produces a specific pattern of numbers. When it looks at a "City," it produces a different pattern. Even though MobileNet has never seen your specific game before, it is very good at detecting the *visual differences* (edges, colors, textures) between them.

### 2. The "Brain": KNN Classifier (Inference)
We use a **K-Nearest Neighbors** classifier to make sense of those Feature Vectors.

*   **Training (Teaching):** When you click a tile and label it "Red Tank on Grass," we take the Feature Vector from MobileNet and store it in a database in your browser along with that label. We are essentially building a "memory bank" of examples.
*   **Inference (Guessing):** When you click "Analyze Map," the AI looks at a new, unknown tile. It converts it to a Feature Vector and calculates the mathematical distance between this new tile and every example in its memory bank.
*   **The Result:** If the new tile's numbers are mathematically closest to the examples you labeled "Water," the AI classifies it as "Water."

### 3. Few-Shot Learning
Because MobileNet is already so smart at seeing visual patterns, we only need to show this tool **1 to 5 examples** of a tile to teach it effectively. This is called **Few-Shot Learning**. You don't need big datasets; you just need a few clear examples.

---

## 🛠 Features

*   **100% Local**: No API keys required. Runs on your CPU/GPU via TensorFlow.js.
*   **Multi-Faceted Classification**: Identifies 5 properties per tile simultaneously:
    1.  **Terrain** (Grass, Forest, Mountain...)
    2.  **Unit** (Tank, Infantry, Mech...)
    3.  **Tile Owner** (Who owns the property/city)
    4.  **Unit Owner** (Who owns the unit standing there)
    5.  **Infrastructure** (Roads, Bridges...)
*   **Interactive Calibration**: precise grid alignment tools with Zoom support.
*   **Knowledge Base Management**: Import/Export your training data to share with others or save for later.

---

## 🚀 Usage Guide

### Step 1: Upload & Calibration
1.  **Upload**: Drag and drop a game screenshot into the left sidebar.
2.  **Zoom**: Use the zoom slider in the right sidebar to get a comfortable view.
3.  **Calibrate Grid**: This is the **most important step**.
    *   Use **Origin X/Y** to align the top-left hex.
    *   Use **Hex Width/Height** to match the size of the tiles.
    *   *Tip:* The yellow dots should be exactly in the center of the hexes, and the green borders should hug the tile edges.

### Step 2: Training (The "Human-in-the-Loop")
The AI starts knowing nothing (or only what you imported). You must teach it.

1.  **Inspect**: Click on a tile in the map. A dialog will open.
2.  **Label**: 
    *   Select the correct **Terrain** (e.g., Plain).
    *   Select the **Unit** (e.g., Tank) or "None".
    *   Select **Owners**. Note: A "Red Unit" can stand on a "Neutral City." These are tracked separately.
3.  **Save & Train**: Clicking save instantly adds this example to the KNN model.

*Repeat this for 1 example of each unique thing you see on the map.*

### Step 3: Analysis
1.  Click the **Analyze Map** button in the header.
2.  The AI will scan every tile coordinates defined by your grid.
3.  **Results**: The right sidebar will populate with predictions.
4.  **Review**:
    *   Click a result in the list to jump to it on the map.
    *   Use **Up/Down Arrow Keys** to quickly review tiles.
    *   If the AI is wrong, click the tile on the map, correct the label, and Save. This "Fine-tunes" the model instantly.

### Step 4: Save Your Brain
1.  In the left sidebar, click **Export KB**.
2.  This downloads a `.json` file containing all your images and labels.
3.  Next time you open the app, click **Import KB** to restore your trained model.

---

## ⌨️ Controls

*   **Map Interaction**: Click to Inspect/Train.
*   **Results List**: Click to highlight/scroll.
*   **Arrow Up/Down**: Navigate through analyzed results.
*   **Scroll**: Pan around the map (enabled when Zoom > 1.0).

---

## 🔧 Technical Stack

*   **Frontend**: React (v18+), TypeScript, Vite/CRA.
*   **Styling**: Tailwind CSS.
*   **ML Core**: TensorFlow.js (Google's ML library for JavaScript).
*   **Models**: 
    *   `@tensorflow-models/mobilenet`: Feature Extractor.
    *   `@tensorflow-models/knn-classifier`: Classification logic.

## 🏗 Development

This project uses a flat file structure for simplicity in AI-generated coding environments, but logically follows:

*   `App.tsx`: Main controller.
*   `services/tfService.ts`: The Brain. Handles all TensorFlow interactions.
*   `utils/hexUtils.ts`: The Math. Handles Hexagonal Coordinate conversion (Axial to Pixel).
*   `components/`: UI building blocks.

To run locally:
1.  `npm install`
2.  `npm start`
