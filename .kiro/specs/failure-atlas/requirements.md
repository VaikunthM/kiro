# Requirements: Failure Atlas (P1)

## Introduction

The Failure Atlas shows what real warehouse pick failures look like and which ones we recreate in simulation. It uses Amazon's ARMBench image-defect dataset (CC BY 4.0): photos of an item held by the suction gripper right after a pick, each labeled clean, multi-pick or package defect, with a sublabel for package defects.

Revision note: the first version of this spec grouped photos with k-means on CLIP embeddings. That grouped photos by camera station, not by failure, so the requirements below use Amazon's own labels instead.

## Requirements

### Requirement 1: Image list from the dataset's own labels

**User story:** As a teammate, I want one list of images with trustworthy labels, so every later step uses the same data.

#### Acceptance criteria
1. WHEN the manifest script runs on the ARMBench folder, THEN it SHALL read each image's sibling `.json` file for its label and sublabel.
2. WHEN a label contains "multi", THEN the image SHALL be labeled `multi_pick`; WHEN it is `nominal`, THEN `clean`; otherwise `package_defect`.
3. THE script SHALL print every raw label and sublabel it saw, so a wrong mapping is visible.
4. THE manifest SHALL contain a balanced sample of at most 3,000 images per class.

### Requirement 2: Failure modes from Amazon's labels

**User story:** As a viewer, I want failures grouped by what actually went wrong.

#### Acceptance criteria
1. WHEN a failure has label multi-pick, THEN its mode SHALL be `multi_pick`.
2. WHEN a failure is a package defect, THEN its mode SHALL be its sublabel (open box, open book, book jacket, open book jacket, partial box), with torn or empty bag grouped as `bag_damage` and everything else as `other_damage`.
3. FOR ANY label and sublabel text, THE mapping SHALL return one of the 8 known modes.

### Requirement 3: Only multi-pick is simulated

**User story:** As the simulator and robot-brain owners, I want to know which failure to build scenes for.

#### Acceptance criteria
1. THE export SHALL mark exactly one cluster, `multi_pick`, as `simulated`.
2. THE simulated cluster SHALL carry exactly three scene modes: `flat_stacked`, `small_beside_big`, `against_wall`.
3. IF the number of simulated clusters is not one, or its scene modes are not three, THEN the export SHALL stop with an error that says what is wrong.
4. THE page and README SHALL state that the scene modes are hypotheses about causes, not findings from the photos.

### Requirement 4: `atlas.json` follows the team contract

**User story:** As the Robot page owner, I want a stable data file.

#### Acceptance criteria
1. THE file SHALL contain `clusters` (id, name, defect, count, examples) and `points` (image, x, y, cluster).
2. EVERY point SHALL belong to an existing cluster, with x and y between 0 and 1.
3. THE cluster counts SHALL add up to the number of points.
4. THE export SHALL write the file to a temporary name, re-read it as JSON, and only then replace the real file.

### Requirement 5: Real-data classifier results

**User story:** As a judge, I want honest numbers on what a standard vision model can detect.

#### Acceptance criteria
1. THE export SHALL train logistic regression on frozen CLIP ViT-B/32 embeddings and test on held-out images.
2. THE results SHALL report three-class accuracy, failure-versus-clean accuracy, per-class recall and per-mode detection rate.
3. THE write-up SHALL state that failure-versus-clean accuracy is only slightly above always answering "failure" on this test set.

### Requirement 6: Atlas page

**User story:** As a viewer, I want to explore the failures.

#### Acceptance criteria
1. WHEN the page loads, THEN it SHALL show a 2D map of failures colored by mode, with no dependency on an external library.
2. WHEN a mode is clicked, THEN the page SHALL show its name, count, share, classifier detection rate and example photos.
3. THE page SHALL credit ARMBench (CC BY 4.0) and say that counts come from a sample.
