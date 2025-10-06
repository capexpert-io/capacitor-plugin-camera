# On-Device Blur Detection Pipeline (Medical Device Labels)

This plan ensures robust blur detection for medical devices and their identification labels/stickers, optimized for **on-device use** with **Google ML Kit** and a **TFLite blur model**.

---

## 🔹 Step 1. Object Detection (Preferred Path)

* Use **Google ML Kit Object Detection** ([docs](https://developers.google.com/ml-kit/vision/object-detection/android)) to detect **labels/stickers** or rectangular regions on the device.
* If **object(s) detected**:

  * Crop ROI (Region of Interest) around detected object(s).
  * Run **TFLite blur detection model** on ROI.
  * Decide blur vs. sharp.

---

## 🔹 Step 2. Text Detection (Fallback if Object Not Found)

* Run **ML Kit Text Recognition** on the full image.
* If text detected:

  * Extract bounding boxes of text blocks.
  * Select **top 3 largest areas** (these are likely stickers/labels).
  * Run **TFLite blur detection model** on each ROI.
  * If **any ROI is blurry**, classify as blurry.
* If no text detected → go to Step 3.

---

## 🔹 Step 3. Full-Image Blur Detection (Final Fallback)

* Run **TFLite blur detection model** on the entire image.
* Use as the final decision when neither object detection nor text detection yields usable ROIs.

---

## 🔹 Decision Flow (Summary)

1. **Object Detection → ROI → Blur Check**
   ✅ Preferred, fastest, most reliable.

2. **No Object → Text Detection → Top 3 ROIs → Blur Check**
   ✅ Works when labels aren’t detected as objects but text exists.

3. **No Object & No Text → Full-Image Blur Check**
   ✅ Ensures no image is left unchecked.

---

## 🔹 Enhancements & Notes

* **Performance tip**: Parallelize blur detection for multiple ROIs to improve speed.
---
