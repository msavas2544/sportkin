# sportkin

[![tests](https://github.com/msavas2544/sportkin/actions/workflows/tests.yml/badge.svg)](https://github.com/msavas2544/sportkin/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23527829.svg)](https://doi.org/10.5281/zenodo.23527829)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Offline, camera-based kinematics, jump analysis and reliability statistics for sport science.**
**Spor bilimleri için çevrimdışı, kamera tabanlı kinematik, sıçrama analizi ve güvenirlik istatistikleri.**

> Status: `0.1.0`, early development. Not yet validated against a criterion
> device; do not use for decisions about athletes until a validation study is published.

---

## English

`sportkin` turns ordinary phone or camera videos into research-grade measurements
and produces the statistics needed for validity and reliability studies. Everything
runs locally: no cloud services, no data leaves the computer.

### Features

| Module | What it does |
|---|---|
| `sportkin.kinematics` | Joint and segment angles (2-D/3-D), zero-lag Butterworth low-pass filter, Winter's residual analysis for cut-off selection, angular velocity, gap interpolation |
| `sportkin.jump` | Jump height from flight time (h = g·t²/8): automatic take-off/landing detection, manual frame correction, one-frame error estimate |
| `sportkin.cmj` | Countermovement jump analysis from a side-view video: height, flight time, peak knee/hip flexion, knee extension velocity, landing flexion |
| `sportkin.pose` / `video` | MediaPipe Pose (33 landmarks) on video files; backend-independent `PoseSeries` with CSV import/export |
| `sportkin.stats` | ICC (all six Shrout & Fleiss forms, 95 % CI), Bland-Altman (bias, LoA, CIs, proportional bias), typical error, CV %, MDC95, SWC, Cohen's d, Pearson r, paired t |
| `sportkin.report` | APA 7 tables (Word/Markdown, English or Turkish) and journal-ready Bland-Altman figures |
| `sportkin.annotate` | Skeleton overlay video with event markers and optional face blurring |

ICC values are validated against the worked example in Shrout & Fleiss (1979) and
against `pingouin`; the jump pipeline is tested on synthetic jumps with known ground
truth (noise, run-up floor drift, landmark sliding after touchdown, variable frame
rate) and was checked frame by frame against a real 30 fps phone video.

Take-off and landing detection: the floor is measured locally before take-off and
after landing, and contact is refined by the impact signature (abrupt stop of the
falling foot), because the toe landmark keeps sliding in the image after touchdown.

### Install

```bash
pip install sportkin            # core: numpy + scipy only
pip install "sportkin[all]"     # + video/pose (OpenCV, MediaPipe) and Word/figure output
sportkin download-model         # once; afterwards everything runs offline
```

Without internet, download `pose_landmarker_full.task` on another computer and pass
`--model PATH` (or set `SPORTKIN_MODEL`).

### Command line

```bash
# analyse a clip; writes CSVs, JSON and an overlay video
sportkin cmj jump.mp4 --out results/ --annotate --blur-face
sportkin cmj slowmo.mp4 --fps 240 --out results/     # slow motion saved at 30 fps playback

# correct take-off/landing by hand (frame numbers from the overlay video)
sportkin cmj --pose results/jump_pose.csv --fps 240 --frames 412:530

# validity: video vs criterion device -> APA table (docx), Bland-Altman figure, JSON
sportkin agreement data.csv --a video_cm --b mat_cm --name-b "Contact mat" --label "CMJ height (cm)"

# test-retest reliability (ICC, TE, CV%, MDC95, SWC)
sportkin reliability data.csv --a trial1 --b trial2
```

Add `--lang tr` for Turkish tables and figures, `--decimal ,` for Turkish Excel CSVs.

### Python

```python
import sportkin as sk

knee = sk.joint_angle(hip, knee_pt, ankle)             # (n_frames, 2) arrays
knee_f = sk.butter_lowpass(knee, fs=240, cutoff=6)

jump = sk.jump_from_frames(takeoff_frame=412, landing_frame=530, fps=240)
print(round(jump.height_cm, 1))                         # 29.6

res = sk.agreement(video_cm, device_cm)                 # ICC, Bland-Altman, d, r, t
```

### Filming protocol (recommended for research)

* Side view, camera on a tripod **perpendicular** to the plane of motion, lens at hip height,
  3–5 m away, whole body visible during the entire jump.
* Capture at **240 fps** (120 fps minimum). Without `--fps`, the real frame timestamps
  are measured from the file (container fps metadata is often wrong). For slow-motion
  files that were saved at playback speed, pass the capture rate, e.g. `--fps 240`.
* Good lighting, plain background, tight clothing; record ≥1 s of quiet standing first.
* Angles are 2-D projections and are only valid for this set-up; report it in the methods.

---

## Türkçe

`sportkin`, sıradan telefon veya kamera videolarından araştırma düzeyinde ölçüm
çıkarır ve geçerlik-güvenirlik çalışmalarının ihtiyaç duyduğu istatistikleri
üretir. Her şey bilgisayarda çalışır, veriler hiçbir yere gönderilmez (KVKK ve etik
kurul süreçleri için önemlidir).

### Ne yapar?

* **Sıçrama (CMJ):** Yan çekim videodan sıçrama yüksekliği, uçuş süresi, çömelme
  derinliği (diz/kalça fleksiyonu), diz ekstansiyon hızı, iniş fleksiyonu.
* **Kinematik:** Eklem açıları, Butterworth filtre, Winter rezidüel analizi, açısal hız.
* **İstatistik:** SİKK/ICC (%95 GA), Bland-Altman, tipik hata, VK%, MSD95, Cohen d,
  Pearson r, eşleştirilmiş t testi.
* **Makale çıktısı:** APA 7 Word tablosu (Türkçe/İngilizce) ve Bland-Altman grafiği.
* **KVKK:** İsteğe bağlı yüz bulanıklaştırmalı işaretli video.

### Kullanım

```bash
pip install "sportkin[all]"
sportkin download-model                                   # bir kez, internet gerekir
sportkin cmj sicrama.mp4 --fps 240 --annotate --blur-face --decimal ,
sportkin agreement veri.csv --a video_cm --b mat_cm --name-b "Kontak mat" --lang tr
```

### Çekim protokolü

* Yandan çekim, tripod, kamera hareket düzlemine **dik**, lens kalça hizasında, 3–5 m mesafe.
* **240 fps** (en az 120). `--fps` verilmezse karelerin gerçek zaman damgaları ölçülür
  (dosyadaki fps etiketi çoğu zaman yanlıştır). Oynatma hızında kaydedilmiş ağır çekim
  dosyalarında çekim hızını verin: `--fps 240`.
* İyi ışık, sade arka plan, dar kıyafet; sıçramadan önce en az 1 sn hareketsiz durma.

---

## Citation / Atıf

If you use `sportkin` in research, please cite it (see `CITATION.cff`):

> Savaş, M. (2026). *sportkin: Offline, camera-based kinematics and reliability statistics
> for sport science* (Version 0.1.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.23527829

The DOI above always points to the latest version; version 0.1.0 is
[10.5281/zenodo.23527830](https://doi.org/10.5281/zenodo.23527830).

## License

MIT
