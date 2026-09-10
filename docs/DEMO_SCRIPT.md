# Judge demo — 2 to 3 minutes

Prepare beforehand: Space open in one tab, 6 test images in `demo/sample_images/`, README metrics table filled in.

| Time | Do | Say |
|---|---|---|
| 0:00 | Show title | "Manual surface inspection is slow and inconsistent. We built a CNN-based system that finds, names and locates defects and issues a PASS/FAIL." |
| 0:20 | Tab 1: upload a `scratches` test image, click Run | "A YOLO object detector — a CNN — draws a bounding box, labels the defect type and gives a confidence score." |
| 0:45 | Point at status | "Any defect above the threshold marks the part FAIL. Defect count is shown for the report." |
| 1:00 | Tab 2: same image, set brightness 0.6 and rotation 30, click | "Real lines have lighting and orientation changes. We perturb the image and re-run. This shows behaviour under controlled variation, not a guarantee for every condition." |
| 1:30 | Tab 3: drop 6 images, click | "Batch mode inspects many parts and produces a CSV report — the automated inspection output the challenge asks for." Click download. |
| 2:00 | README metrics table | "On the NEU-DET validation set the model reached mAP50 of ___ and recall of ___." (read your real numbers) |
| 2:20 | Close | "Next step in a factory: a camera over the line feeding this model, results pushed to the plant system, threshold tuned to the plant's acceptance criteria." |

# Likely questions

**Why YOLO?** Single-pass detector, fast enough for a moving line, mature training tooling, pretrained weights so it trains in minutes on a small dataset.

**Why CNN?** Defects are local texture patterns. Convolutional filters detect edges and textures regardless of position, which is exactly the problem.

**Why detection instead of classification?** The challenge asks *where* the defect is and to count defects. A classifier gives one label per image and no location.

**Dataset?** NEU-DET, hot-rolled steel strip, six classes, 200×200 images, downloaded in YOLO format from Roboflow Universe. Split sizes: (from notebook section 4).

**How many classes?** Six: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

**How did you train?** Google Colab T4 GPU, Ultralytics YOLO26n pretrained on COCO, 256 px, batch 32, up to 60 epochs with early stopping, default augmentation. A 1-epoch smoke test first.

**What is confidence?** The model's probability estimate that a box contains that defect. We hide boxes below 0.25.

**Precision and recall?** Precision: of the boxes we drew, how many were real. Recall: of the real defects, how many we found. Trade-off controlled by the threshold.

**How do you reduce false positives?** Raise the confidence threshold, train longer with more varied data, add hard-negative images of defect-free surfaces, tune the threshold on the validation precision/recall curve (notebook 8c).

**How do you handle lighting variation?** Training augmentation randomises brightness/contrast (HSV) and geometry, so the model sees varied conditions. The Robustness tab demonstrates it. Real deployment would also fix the lighting rig.

**How do you know it works?** Validation metrics on held-out images (read them). We also ran it on test images it never saw. We have *not* tested on non-NEU-DET images.

**Limitations?** One dataset, one steel type, grayscale 200 px images; six classes only — cracks and dents are not in NEU-DET; demo PASS/FAIL rule; no defect-size measurement.

**Real factory deployment?** Industrial camera + controlled lighting → edge PC running the exported model (ONNX/TensorRT) → PASS/FAIL to a PLC, results logged to the plant database. Retrain periodically on newly labelled factory images.

**Live camera?** Yes — Ultralytics supports `model.predict(source=0)` for webcams and RTSP streams. Not demonstrated today.

**Integration with a manufacturing system?** Publish each inspection result (image ID, status, defects, confidences) over MQTT or a REST call to the MES; the CSV report is the offline version of the same record.

**What would you do with one more week?** Collect real factory images, measure defect sizes in mm via camera calibration, tune the threshold per class, add a "review queue" for low-confidence cases.