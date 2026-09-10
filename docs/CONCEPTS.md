# Concepts — explained with our project

Read this once before training. Every term is defined using our steel-defect system.

**CNN (Convolutional Neural Network)** — a neural network that slides small filters over an image to find edges, textures and shapes. Early layers learn "dark thin line", later layers learn "scratch". YOLO is built from CNN layers, so our detector is a CNN.

**Classification** — answering *what* is in the image: "this image contains a scratch". One label for the whole image.

**Localization** — answering *where*: "the scratch is in the top-left region". Given as coordinates.

**Object detection** — classification + localization for every object: "scratch at (x1,y1,x2,y2) with 0.87 confidence, inclusion at (…) with 0.62". This is what the challenge asks for, and why we use a detector instead of a plain classifier.

**Bounding box** — a rectangle around a defect, stored as four numbers. YOLO labels use `x_center y_center width height` normalised to 0–1; the app shows them as pixel corners `x1 y1 x2 y2`.

**YOLO (You Only Look Once)** — a detector that predicts all boxes in one pass of the network. Fast enough for a production line. We use Ultralytics YOLO26n (nano = smallest).

**Dataset** — the images plus their correct answers. Ours is NEU-DET: steel surface images with boxes around six defect types.

**Annotation / label** — the correct answer for one image: a `.txt` file with one line per defect, `class_id x y w h`. The **class ID** is a number (0–5) that maps to a name (`crazing`, `inclusion`, …) in `data.yaml`.

**Training** — showing the model labelled images thousands of times. Each time, it predicts boxes, measures how wrong it was (loss), and nudges its internal numbers (weights) to be less wrong.

**Epoch** — one full pass over every training image. We run up to 60.

**Model weights** — the millions of numbers inside the network that training adjusts. Saved as a `.pt` file.

**best.pt** — the weights from the epoch with the highest validation mAP. `last.pt` is simply the final epoch. We ship `best.pt`.

**Inference** — using the trained model on a new image to get predictions. The Gradio app does inference; it never trains.

**Confidence** — the model's estimated probability (0–1) that a box really contains that defect. We hide boxes below a threshold (default 0.25, a demo value).

**False positive** — the model draws a box where there is no defect → a good part is rejected (waste).
**False negative** — a real defect gets no box → a bad part ships (customer complaint). In factories false negatives usually cost more.

**Precision** — of all boxes the model drew, the fraction that were real defects. High precision = few false alarms.
**Recall** — of all real defects, the fraction the model found. High recall = few missed defects.
Raising the confidence threshold raises precision and lowers recall; lowering it does the opposite.

**mAP50** — mean Average Precision when a predicted box counts as correct if it overlaps the true box by at least 50% (IoU ≥ 0.5). Averaged across the six classes. Our headline metric.
**mAP50-95** — the same, averaged over IoU thresholds 0.50, 0.55, … 0.95. Stricter; rewards tightly fitting boxes. Always lower than mAP50.

**Training vs validation vs real-world performance** — training metrics are measured on images the model has seen (optimistic). Validation metrics use held-out images (honest estimate of how it generalises within NEU-DET). Real-world performance on your factory's camera and lighting is unknown until you test it there.