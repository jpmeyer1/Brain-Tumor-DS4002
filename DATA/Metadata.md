## Data Summary
A brain tumor detection dataset consists of medical images from MRI or CT scans, containing information about brain tumor presence, location, and characteristics. This dataset is essential for training computer vision algorithms to automate brain tumor identification, aiding in early diagnosis and treatment planning in healthcare applications.

## Provenance
The Ultralytics Brain‑Tumor Dataset is a publicly documented medical-imaging dataset curated by Ultralytics. It comprises 893 labeled training images and 223 labeled test images drawn from MRI or CT brain scans, with binary class annotations for “negative” (no tumor) and “positive” (tumor present). The dataset is supplied in a standardized YOLO-formatted structure (images and corresponding annotation files) and accompanied by a YAML configuration specifying dataset paths and class names. The dataset is intended for research and development of real-time object-detection models in clinical and diagnostic contexts, such as early tumor identification and treatment-planning support.

## License
This software (and any models derived from it) is licensed under the GNU Affero General Public License v3 (AGPL-3.0) via Ultralytics. Under this licence, you are free to use, share and modify the code (including for commercial purposes), provided that if you distribute it, you also must make the full source of your version available under the same licence. If you do not want to open-source your modifications or derived work, you must obtain a separate commercial licence from Ultralytics. 

## Ethical Statements
The Ultralytics Brain Tumor dataset raises important ethical considerations related to privacy, patient consent, and clinical safety. Because MRI scans contain highly sensitive medical information, all data must be fully de-identified and handled in compliance with HIPAA standards. Patients whose scans are included should have provided informed consent before any public or commercial release. From a clinical perspective, false negatives pose the greatest risk, as missed tumors could delay treatment and endanger patient outcomes, while false positives may cause unnecessary stress or medical intervention.

## Data Dictionary
| Column          | Description                                                        | Potential Responses / Examples     |
| --------------- | ------------------------------------------------------------------ | ---------------------------------- |
| **image**       | JPG image of the brain                                             | *(image)*                          |
| **image title** | Name of the brain image file, matched with labels of the same name | `"00054_145.jpg"`, `"62 (13).jpg"` |
| **label title** | Name of the associated `.txt` label file                           | `"00054_145.txt"`, `"62 (13).txt"` |
| **class**       | Classifier for the datapoint: **0 = negative**, **1 = positive**   | `0`, `1`                           |
| **x_center**    | X-coordinate of the center of the tumor box (normalized 0–1)       | `0.344484`, `0.518779`             |
| **y_center**    | Y-coordinate of the center of the tumor box (normalized 0–1)       | `0.342723`, `0.416667`             |
| **width**       | Width of the bounding box (normalized 0–1)                         | `0.221831`, `0.150235`             |
| **height**      | Height of the bounding box (normalized 0–1)                        | `0.176056`, `0.070423`             |


## Exploratory Plots
![Tumor Class Distribution](Tumor%20Class%20Distribution.png)
![Bounding Box Distribution](Bounding%20Box%20Distribution.png)
