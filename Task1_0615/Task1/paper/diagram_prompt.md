# Prompt for ChatGPT/DALL-E: Task 1 method architecture diagram

Paste this directly:

---

Create a clean, professional academic flowchart diagram (white background,
simple rectangular boxes, thin black outlines, sans-serif labels, arrows
showing data flow left-to-right) illustrating a medical AI model pipeline
for chest X-ray tuberculosis cavity classification and segmentation. This
is for a scientific paper figure -- no photorealistic elements, no
shading/gradients, no decorative icons, just a clean technical block
diagram like you'd see in a computer vision paper.

Layout, left to right:

1. Leftmost box: "Raw Chest X-ray".

2. Arrow to a box labeled "4-Member Classification Ensemble\n(DenseNet-201,
   EfficientNetV2-S, ResNet-50D,\nOrdFused/EVA-X + CORN ordinal head)".

3. Arrow to a box labeled "Probability Average\n+ Threshold tau*=0.535".

4. Arrow SPLITS into two branches based on the classifier's decision:

   Branch A (top, labeled "cavity-negative"): straight arrow to a box
   labeled "Empty Mask (no cavity)".

   Branch B (bottom, labeled "cavity-positive"): arrow to a box labeled
   "3-Member Segmentation Decoder Ensemble\n(RAD-DINO, CheXFound ViT-L/16,
   EVA-X-base)\nFrozen backbone + LoRA, Focal Tversky + BCE loss" -> arrow
   -> a box labeled "Cavity Mask".

5. Both branch outputs ("Empty Mask" and "Cavity Mask") converge into a
   final rightmost box, drawn slightly larger/bolder outline: "Final
   Prediction: pred_cavity + mask".

Keep all text horizontal and legible, use a consistent box size within each
column. Title at the top of the image, centered: "Classifier-Gated
Foundation-Model Ensemble for TB Cavity Detection".

---

## Notes if the generated image looks off

Text-to-image models (DALL-E, etc.) are unreliable at rendering exact text
inside boxes -- labels often come out garbled or misspelled. If that
happens, two better options:

1. **Ask ChatGPT to generate Mermaid or a simple SVG/TikZ code instead of an
   image** -- e.g. "generate this as a Mermaid flowchart" or "as a TikZ
   diagram for LaTeX" -- this renders crisp, correct text every time and
   drops straight into the paper (TikZ compiles natively in LaTeX/Overleaf,
   no image file needed at all).
2. Use a free diagramming tool (draw.io / Excalidraw) with the same layout
   described above -- a few minutes of manual drag-and-drop but guarantees
   correct labels.

Given the deadline, option 1 (ask for Mermaid or TikZ code) is probably
faster and more reliable than iterating on image-generation prompts.
