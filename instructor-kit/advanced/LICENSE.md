# Instructor Kit License (Summary / Template)

**Course:** Python Machine Learning · **Kit:** Advanced Machine Learning (Modules 7 to 13, 13 sessions)
(<https://rays-python-machine-learning.netlify.app/>)
**Author & licensor:** Ray de la Paz

> This is a plain-language template. Replace the bracketed terms at point of sale
> and have final wording reviewed by a lawyer before commercial distribution,
> including the "Third-party data and models" section below.

## Licensed materials
"The kit" means everything in the `instructor-kit/advanced/` folder and the course data it is taught with:

- the facilitator guides, participant workbooks and answer keys (per session and combined), the slide
  decks, the labs (starters, solutions, debug handouts, fixtures and checkers), the setup guide, the
  final assessment and the sell sheet;
- the **Pinecrest Outfitters dataset** (`pinecrest-ml-data.zip`: the seven CSV files, `README.md` and
  the generator `make_ml_data.py`). Pinecrest Outfitters is fictional; the dataset is an original work
  and part of the licensed materials when used with the kit.

The kit does **not** include, and this license does **not** cover, the third-party image datasets and
pretrained models listed below. Learners download those from their original sources.

This license covers the Advanced Machine Learning kit only. The companion kit, Machine Learning
Foundations, is licensed separately.

## Grant
You (the "Instructor") are granted a **non-exclusive, non-transferable license** to
use the kit to **teach the course to your own students**, according to the
tier you purchased:

| Tier | Who may teach | Rebranding | Typical price |
|------|---------------|------------|---------------|
| **Solo** | One named instructor | No | [$1,497] one-time |
| **Organization** | Multiple instructors at one org | No | [$4,997] one-time |
| **White-label** | Multiple instructors at one org | Yes — edit & add your brand | [$9,997] one-time |
| **Updates add-on** | — | — | [$299 / yr] |

## You MAY
- Teach the course to an unlimited number of your own students under your tier.
- Print and distribute the **participant workbook**, the **setup guide** and the **lab starter files**
  (with `labs_paths.py` and the `_fixtures/` folders) to enrolled students.
- Give enrolled students the **Pinecrest dataset** (`pinecrest-ml-data.zip`).
- Hand out lab **solutions** to enrolled students after the lab has been completed in class.
- Adapt timing, examples, and pacing to your audience.
- (White-label only) Edit the materials and add your own branding.

## You MAY NOT
- Resell, sublicense, or redistribute the kit itself to other instructors.
- Distribute the **facilitator guide**, the **answer key** or the **final assessment** (its rubric
  and instructor answer notes) to students. These are **confidential** instructor materials.
- Publish the materials publicly (e.g., a public website, repository or file share), including
  republishing the Pinecrest dataset as your own.
- Remove author credit (except under the white-label tier).

## Third-party data and models
Four lessons (9.3 and 10.3 to 10.5) and their labs use two public image datasets and two pretrained
networks. They belong to their authors, not to the licensor. **The kit does not redistribute the
datasets or the weights**: learners download them from their original sources with torchvision
(`check_environment.py --download`, or the lessons' own `download=True` and `weights=` calls). The
statements below were checked against the sources in October 2026; check them again before you sell.

| Item | Used in | Owner and license | What it means for you |
|------|---------|-------------------|-----------------------|
| **Fashion-MNIST** (70,000 grayscale clothing images) | Lesson 9.3, Module 10, Sessions 6 to 8 | Zalando Research; **MIT License**, "Copyright © 2017 Zalando SE" (<https://github.com/zalandoresearch/fashion-mnist>) | Free to use for teaching, including commercially. Keep the copyright and license notice with any copy. |
| **EuroSAT RGB** (27,000 Sentinel-2 satellite tiles, 10 land-cover classes) | Lesson 10.5, Session 9 | Helber, Bischke, Dengel and Borth (DFKI); the authors state "The dataset is licensed under the MIT license" (<https://github.com/phelber/EuroSAT>). The images are derived from Copernicus Sentinel-2 data, which is free and open; the authors ask users to consider the Copernicus Sentinel Data Terms and Conditions and to cite their two papers (IEEE JSTARS 2019; IGARSS 2018). | Free to use for teaching. Credit the authors and "contains modified Copernicus Sentinel data" when you show the images in your own materials. |
| **torchvision** (the code that downloads the data and builds the networks) | Modules 9 and 10 | PyTorch contributors; **BSD 3-Clause License** | Free to use, including commercially. |
| **Pretrained weights: ResNet-18 (`IMAGENET1K_V1`) and MobileNetV3-small (`IMAGENET1K_V1`)** | Lesson 10.5, Session 9 (and its lab fixture) | Published by the torchvision project; **trained on ImageNet**. The torchvision documentation says pretrained models "may have their own licenses or terms and conditions derived from the dataset used for training" and that "it is your responsibility to determine whether you have permission to use the models for your use case." ImageNet's terms of access allow use of the database "only for non-commercial research and educational purposes." | The weights are used here for teaching, which is an educational purpose, but the legal status of models trained on ImageNet is not settled and the terms say "non-commercial". **If you teach for profit (a paid course, a bootcamp, corporate training), review these terms with your own counsel before you rely on the pretrained weights.** If in doubt, the course works without them: Lesson 10.5's from-scratch CNN and the Fashion-MNIST sessions use no pretrained weights. |

**Derived files in the kit.** Two lab fixture folders hold small files computed from these sources,
so the labs run quickly and offline: `labs/module-08/_fixtures/` (two small networks trained on
Fashion-MNIST, MIT) and `labs/module-09/_fixtures/eurosat_resnet18_feats.npz` (2.4 MB of
ResNet-18 features of EuroSAT tiles: derived from EuroSAT, MIT, and from the ImageNet-trained
weights above). They contain no images. The same caution about the ImageNet-trained weights applies
to the EuroSAT features file; `prepare_eurosat_features.py` rebuilds it on your own machine if you
prefer not to distribute it.

Required notices for the MIT-licensed datasets, to keep with any copy of the derived files:

```
Fashion-MNIST: The MIT License (MIT). Copyright (c) 2017 Zalando SE, https://tech.zalando.com
EuroSAT: MIT License. P. Helber, B. Bischke, A. Dengel, D. Borth. EuroSAT: A Novel Dataset and Deep
Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS, 2019.
Contains modified Copernicus Sentinel data.
```

## The public course site
The lessons at <https://rays-python-machine-learning.netlify.app/> and the public syllabus remain
freely available to anyone. This license covers the instructor kit and its use for teaching; it does
not restrict anyone's use of the free lessons.

## Ownership
All materials remain the intellectual property of the licensor, except the third-party datasets,
code and models named above, which remain the property of their owners. This license may be
revoked for breach of these terms.

---
© 2026 Ray de la Paz. All rights reserved.
