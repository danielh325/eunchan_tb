# Prompt for ChatGPT/DALL-E: Task 2 method architecture diagram

Paste this directly:

---

Create a clean, professional academic flowchart diagram (white background,
simple rectangular boxes, thin black outlines, sans-serif labels, arrows
showing data flow left-to-right) illustrating a medical AI model pipeline
for chest X-ray tuberculosis classification. This is for a scientific paper
figure -- no photorealistic elements, no shading/gradients, no decorative
icons, just a clean technical block diagram like you'd see in a computer
vision paper (similar style to a standard "model architecture" figure).

Layout, left to right:

1. Leftmost box: "Raw Chest X-ray" with a small simple chest X-ray icon or
   just the text label.

2. Arrow to a box labeled "Lung-Field Segmentation & Crop (PSPNet)".

3. Arrow to a box labeled "CLAHE + Intensity Normalization".

4. Arrow from there SPLITS into two parallel horizontal branches (draw both
   branches starting from the same point, stacked vertically, each flowing
   left to right in parallel):

   Branch A (top): a box labeled "RAD-DINO (ViT-B/14)\nFrozen backbone + LoRA
   adapters" -> arrow -> a small box labeled "4-fold ensemble\n(leave-one-
   modality-out)" -> arrow -> a box labeled "P(TB) probability".

   Branch B (bottom): a box labeled "CheXFound (ViT-L/16)\nFrozen backbone +
   LoRA adapters" -> arrow -> a small box labeled "4-fold ensemble\n(leave-
   one-modality-out)" -> arrow -> a box labeled "P(TB) probability".

5. Both "P(TB) probability" boxes from branch A and branch B converge with
   arrows into a single box labeled "Probability Averaging".

6. Final arrow to the rightmost box, drawn slightly larger/bolder outline:
   "TB / Normal Decision (threshold = 0.5)".

Keep all text horizontal and legible, use a consistent box size within each
column, and keep the whole diagram in a single row-based flow (top branch
and bottom branch same height, symmetric). Title at the top of the image,
centered: "Two-Encoder Foundation Model Ensemble for TB Screening".

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
