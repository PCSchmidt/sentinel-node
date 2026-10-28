# Sentinel Node: Evidence Notes

Companion to sentinel-node.md. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author. Depth: LITE. Repo state read: main at the last of 5 commits (b6ceaee, 2026-10-16). Read-only clone; all runs were in a scratch copy made with `git archive`, in a throwaway virtual environment.

## E1 Fact sheet

- Name: sentinel-node; public GitHub repo (PCSchmidt/sentinel-node); MIT licence (Confirmed, LICENSE and GitHub metadata). 0 stars, 0 forks. Created 2026-10-16 19:08Z, last push 2026-10-17 03:25Z.
- Kind: edge-sensor scaffold (harness) plus one GPU experiment. Status per README: "scaffold plus a first detector result" (Per the docs).
- Languages and size: Python only; about 2,300 lines across source, tests, scripts and docs (Confirmed, `wc -l`). Source package is about 600 lines in 6 modules; tests 358 lines; experiment scripts 640 lines.
- Tests: 42 collected, 42 passed in a scratch copy (Ran it): `python -m pytest -q -p no:cacheprovider`, Python 3.14.6, result "42 passed". 39 test functions plus parametrised cases.
- Commits: 5, all on 2026-10-16 (15:03 to 23:25 -0400). All five carry an AI co-author trailer (Confirmed). No agent instruction files or settings folders in the repo (Confirmed, file listing).
- Clean run: `pip install -e ".[dev]"` was not run (nothing installed into the clone); with `PYTHONPATH=src` the CLI ran: `python -m sentinel_node.cli --frames 50` printed "50 frames, 44791 bytes published", and a stage table with totals about 0.16 ms (Ran it). Those timings describe the Python harness only.
- Needs: harness needs nothing beyond Python (dependencies list is empty; only `pytest` for dev). Experiment needs an NVIDIA GPU, PyTorch with CUDA, Ultralytics, ONNX, TensorRT below version 11, and a roughly 2 GB dataset download (Per the docs). Not run here (GPU, large download).
- Deployed: no. Outside users: none known (Ask the author).
- Related: part of physical-ai-portfolio; tracker ideas intended to reuse FuseTrack (Per the docs).

## E2 Reading and running log

Read: README, docs/design.md, docs/roadmap.md, experiment README, all source modules, all five test files (three in full), CI file, pyproject, .gitignore, experiment scripts (train, evaluate, make_calib, calibration, common, part of calib_study), all result JSON/markdown files (spot-parsed with a script), training curve, git log. Not read: summarize.py and latency_repeat.py in detail. Not run: any GPU, training, TensorRT or download step; any `.env` (none exists). Commands run: pytest in the scratch copy, the CLI, a probe script (E4). Final `git status` in the clone: clean (no tracked changes; the clone was never written). Network: `gh` queries for run list, repo metadata and code-scanning alerts only.

## E3 Claims check

Levels: L1 doc matches saved result file; L3 recomputed from saved files.

- CL-01 README table "FP32 0.162 / FP16 0.162 (+0.05) / INT8 0.152 (-0.92); 6.59 / 1.12 / 1.07 ms; 5.2 / 8.1 / 4.9 MB". results.json: 0.16174, 0.16225, 0.15249; 6.59, 1.12, 1.07; 5.23, 8.11, 4.91. Supported, L3 (deltas recomputed).
- CL-02 "FP16 +0.11 ± 0.05 over three models; INT8 MinMax -0.88 ± 0.07". Recomputed from results.json, results_s1.json, results_s2.json: FP16 deltas 0.05, 0.16, 0.12 (mean 0.11, sample std 0.056); MinMax -0.92, -0.92, -0.80 (mean -0.88, std 0.069). Supported, L3.
- CL-03 "Entropy lost about 3 points". study_calibration.json gives -4.19, -2.30, -2.39 (seed-0 model, three calibration samples; mean -2.96, std 1.06 per summary_0_3b.md), and -3.13, -3.06 for trained seeds 1 and 2. Supported, L1 plus L3 for the two seed rows. Note summary_0_3b.md says Entropy "n=2" for seed rows, matching.
- CL-04 "MinMax about 8% faster than FP16 at batch 1 and about 27% at batch 8 (seed-0 model)". latency_repeat_batch8.json medians: FP16 0.819 ms, INT8 MinMax 0.525 ms (ratio 0.64 by medians; the 0.73 figure is the same-round median ratio, stated in docs). The README quotes same-round ratios (0.92 and 0.73). L1 only for the ratios; recomputing medians gives a larger gap (36%) than the same-round ratio, a reminder the ranges overlap. Partly checked; the guide uses the docs' same-round ratios and states drift.
- CL-05 "first run said INT8 slower (1.52 vs 1.05 ms); did not reproduce". results_first_pass.json: 1.5186 and 1.0454 ms. Supported, L3.
- CL-06 "message under 600 B" and 298 B in the guide. Ran it: the sn-1 example is 298 bytes; a test asserts < 600. Supported.
- CL-07 Governor thresholds (20,000 bps for tracks; 4,000 for reduced; hysteresis). Ran it: capacity 20,000 gives TRACKS, 19,000 REDUCED, 21,000 stays REDUCED after a degrade, 27,000 climbs to TRACKS (about 26,667 by arithmetic). Supported, L2.
- CL-08 "Nothing has run on a Jetson." No Jetson code, no device results anywhere. Supported (absence).
- CL-09 Training time "about 52 minutes": training_curve.csv time column ends at 3,125.7 s at epoch 30. Supported; guide's figure is derived, not stated in docs.
- CL-10 Experiment README: "final training-time validation mAP50 0.299, mAP50-95 0.165". training_curve.csv epoch 30: 0.29931 and 0.16473. Supported, L3.

Count: 10 claims checked, 0 contradicted. One doc-versus-doc difference (E14).

## E4 Rule attacks (scratch copy)

Probe script ran against the scratch copy's package. Results (Ran it):
- Latitude NaN and 91: both raise ValueError ("latitude out of range"). Longitude and time-zone-naive datetime are covered by existing tests.
- Identifier containing `<`, `&` and a quote: output is escaped (`a&lt;b&amp;&quot;c`), message still well-formed.
- Governor: capacity 0 and -5 with 5 tracks: OFF. Capacity 0 with zero tracks: FULL (cost is 0, which fits any budget; harmless, surprising). Threshold exactness: 20,000 gives TRACKS; 42,857 TRACKS and 42,858 FULL (30,000 bps needed / 0.7).
- Hysteresis from a REDUCED state: 21,000 and 26,000 stay REDUCED; 27,000 climbs to TRACKS; then 0 drops to OFF at once.
- A new track in the placeholder tracker reports speed 0 and course 0 (atan2(0, 0) = 0), by reading the code (My reading).
- Flat-earth conversion: 3 km east at latitude 38.9 is 0.0346 degrees of longitude by the code's formula; its error against a true geodetic conversion was not measured.
- Not attacked: crash and restart, dropped frames, concurrent use, hostile network input (no network code exists).

## E5 Code map

- src/sentinel_node/cot.py: CoT 2.0 encoder (`to_cot_xml`, `GeoTrack`), range checks, timezone-aware UTC times.
- governor.py: `required_bps`, `choose_level`, stateful `BandwidthGovernor` with hysteresis.
- bench.py: `StageTimer`, `percentile` (linear interpolation), `joules_per_frame`, `parse_tegrastats_power` (regex for the VDD_IN field), `to_markdown`.
- pipeline.py: stage Protocols, synthetic source and detector (seeded `random.Random`), `NearestNeighbourTracker`, `CotPublisher` (flat-earth offsets, sink callback, no transport), `run`.
- cli.py: entry point. detector_report.py: result records, comparison and table, `mean_std`, JSON save and load.
- governor.py is not imported by pipeline.py or cli.py (grep, Confirmed). detector_report.py is imported only by the experiment scripts.
- experiments/detector: train.py (Ultralytics YOLO11n, seed argument, `deterministic=True`, 640 px, 30 epochs by default), make_calib.py (random 512-image sample of the training split), evaluate.py (FP32, FP16, INT8 engines; validator at batch 1; latency by median of rounds), calibration.py (temporary patch of an Ultralytics internal to switch MinMax to Entropy), calib_study.py, latency_repeat.py, summarize.py, results/.

## E6 Model call sites

One learned model: YOLO11n (Ultralytics), COCO-pretrained, fine-tuned on VisDrone2019-DET for 30 epochs, batch 16, seed 0, 1 or 2 (train.py, evaluate.py). No language-model or hosted-service calls anywhere; no API keys needed. Cost: local GPU time only; training about 52 minutes (CL-09). The model is not called from the harness. Model provenance and the licence of the pretrained weights: Ultralytics is AGPL-3.0 (Per the docs, experiment README); the weights file's own licence terms are not in the repo (Ask the author). VisDrone's licence terms are not in the repo.

## E7 Tests and CI

- 42 tests: bench 9 definitions, cot 7 (one parametrised with 4 cases), detector_report 7, governor 8, pipeline 8 (Ran it). All unit tests with in-process data; one end-to-end test uses the synthetic stand-ins. Coverage not measured.
- Spot checks: test_known_thresholds asserts governor levels at 19,000, 21,000, 3,900 and 4,100 bps with the arithmetic in comments (checked against the code by hand and by probe); test_governor_hysteresis_prevents_flapping asserts staying at REDUCED at 21,000 and climbing at 40,000; test_message_is_small asserts length < 600. All assert hand-derived values, not code output (My reading).
- CI (.github/workflows/ci.yml): on push to main and on pull requests; Ubuntu; Python 3.10 and 3.12; installs with `pip install -e ".[dev]"`, runs `pytest -q`, then the CLI for 50 frames. `gh run list`: all 5 CI runs on file are success (each 13 to 16 s), latest on 2026-10-17 03:25Z. A Dependency Graph run (automatic) also succeeded. No lint, type or coverage step; no GPU step. The README badge links to this workflow.
- Tests do not touch the network, a GPU or Docker.

## E8 Plan-versus-results and history

All 5 commits are on one day. Sequence: scaffold 15:03; roadmap commit "do detector INT8 accuracy on a desktop GPU before the Jetson" 15:09; milestone 0.3a results 17:01; correction and calibration studies 19:28; milestone 0.3b 23:25. The roadmap preceded the first result by about 1 h 52 min, but it states no success rule or locked protocol, so order cannot be taken as pre-registration. The GitHub events list was not needed (repo is under 90 days old). Commit credits: 5 of 5 have an AI co-author trailer; the guide names no tool, per the author's standing instruction. The history shows one self-correction (commit 2d7c6ef corrects a claim from 41ae232 about INT8 latency).

## E9 Gates and approvals

None. No gate files, approvals or review records.

## E10 Environment

Dependencies: none in pyproject for runtime; dev needs pytest 8 or later. No lock file. Experiment pins `tensorrt-cu12<11`; results record GPU, torch 2.11.0+cu128, TensorRT 10.16.1.11, Ultralytics 8.4.178 in the result JSON meta. A fresh install of the harness works (Ran it, via PYTHONPATH). A fresh GPU install was not tried. Note: results JSON metadata contains absolute local paths from the author's machine (not reproduced in the guide). The TensorRT build-to-build spread of about 0.14 points for identical inputs is recorded in the experiment README.

## E11 Security and privacy

Secret scan of current files and history: no matches for key, token or password assignments, or common key prefixes (Ran it, `git grep` and `git log -p` pattern search). `.env` is git-ignored; none present. Outside services: GitHub only; the experiment downloads a public dataset. GitHub code-scanning returned "no analysis found". Dependabot's dependency-graph job runs. No network sending code exists, so no attack surface beyond file and package handling. No claim is made about live security state.

## E12 Shortcomings register

Full records for the three that matter; others in the table.

1. Detector not integrated (Gap, High for the project's purpose). Evidence: pipeline uses SyntheticDetector; no import of any model. Impact: no end-to-end detection result. Options: (A) wrap the trained engine behind the Detector interface and run it on recorded video; (B) leave experiment and harness separate and say so. Recommendation: A, since it costs little and gives the stage table a real row. Effort small to medium; payoff high. Verify: stage table has a detect row measured with the real engine.
2. Governor not wired into the run (Gap, Medium). Evidence: grep (E5). The design notes' data-flow diagram shows it between tracker and publisher. Options: (A) call it in the driver and test a degrading link; (B) correct the diagram. Recommend A. Effort small; payoff medium.
3. No device or client validation (Gap, High, disclosed). Nothing run on a Jetson; CoT not checked against ATAK or WinTAK (Per the docs). Options: rent or borrow a device and run engines there; or validate CoT in a free desktop client first. Recommend the client check (cheaper), then device. Effort medium to large; payoff high.

Minor: design.md says dataset choice not made (stale); no power source implementation; flat-earth error unmeasured; validation images double as checkpoint selection and reporting (My reading, standard training-tool behaviour); entropy calibration uses a narrow patch of a third-party internal that may break on other versions (disclosed); zero-track governor reports FULL at zero capacity; first-seen tracks report heading 0.

## E13 Method detail

- Governor: required_bps = tracks x rate x bytes x 8, plus for FULL one 4,000-byte thumbnail per track every 10 s. Level fits if required <= 0.7 x capacity. Upgrade requires fit at 0.7 x 0.75 x capacity. Example: 5 tracks, 1 Hz, 350 B: TRACKS 14,000; REDUCED (0.2 Hz) 2,800; FULL 30,000 bps.
- Percentile: linear interpolation on sorted samples. End-to-end latency is the sum of mean stage latencies; valid because stages run sequentially (design.md).
- Energy: joules = watts x ms / 1000.
- INT8 toy example: scale = 0.90/127 = 0.007087; 0.12 to 17, -0.57 to -80, 0.90 to 127; back: 0.1205, -0.5669, 0.9000.
- mAP50-95: from Ultralytics' validator, not reimplemented. Mean and std: sample std (n-1) in detector_report.mean_std.
- TensorRT 11 removed the classic INT8 calibrator, so the experiment pins version 10 (Per the docs).

## E14 Discrepancies

- design.md "Open decisions: Detector ... Dataset choice is not made" versus README and experiment (VisDrone chosen and used). Stale doc.
- design.md data-flow places the governor between tracker and publisher; code does not call it. README module table is accurate ("capture, detect, track, publish").
- design.md data-flow says "detect (TensorRT INT8)"; the harness uses a synthetic detector. The diagram is the target, not the build.
- README results sentence "about 8% faster than FP16 at batch 1" uses same-round ratios; the median-based comparison in the saved table is closer to 5% (1.07 versus 1.12 ms). Both in the docs.
- Experiment README says the MinMax study mean is -0.79 and the main table -0.92 for the same weights and calibration sample 0; explained there as build-to-build variation of about 0.14 points.

## E15 Production readiness

Placement: harness at "works locally"; detector work at "experiment". Checklist: clean-start documented (Partly: harness yes, GPU environment documented but not run here); checks block a bad change (Partly: CI runs tests, no quality evaluation gate); quality measured against a baseline (Partly: precision comparison against FP32, no outside baseline); safe failure behaviour (No: nothing to fail yet); human review (Not applicable); versioning of prompts and models (Not applicable for prompts; model versions recorded in result files, weights not committed); logs and monitoring (No); data and secrets handled safely (Partly: no secrets, public data, no review). Gap to deployable: hosting, device, transport, tracker, power measurement, monitoring, rollback, security review: all missing. Reusable assets: the CoT encoder, the governor, the timer harness and the experiment scripts are small and reusable in a larger edge project (My reading).

## E16 Review dispositions

Self-review passes applied to the draft: removed first-person claims of checking; replaced an unverified size comparison for video; removed unverified size of the Jetson; softened "free" for FP16; named SSD and RT-DETR only as alternatives not tested. No external review yet.

## E17 Completeness and backlog

Promised versus built: CoT encoder (done and tested); governor (done and tested, not wired); timers (done and tested; no power source); pipeline driver (done with stand-ins); detector accuracy experiment (done, three seeds, GPU only); CoT client validation (planned); UDP and MQTT publishers (planned); Jetson capture, engines, latency and joules (planned); on-node tracker and geo-registration (planned); operations layer, 24 h soak and link tests (planned); two-node cross-cueing (stretch, planned). Backlog groups: Finish: governor into the driver; real detector into the harness. Fix: refresh design.md. Improve: report a fixed-clock latency run. Add: transport, client check, Jetson runs. Next milestone: roadmap 0.4 (engines on a Jetson, with latency and joules per frame at 7, 15 and 25 W). Whether to do these is for the author to decide.

## E18 Revision log (second draft)

Archived the first draft as sentinel-node.v1draft.md and applied the ten reader-check fixes. Length 3,920 to about 3,520 words (plain wc -w).
1. "Harness" defined at first use as a test rig; the guide then says "test rig".
2. GPU: every saved result file records "NVIDIA RTX A4500 Laptop GPU" (Confirmed, results.json). The repo README says "desktop GPU" meaning "ordinary graphics card, not a Jetson"; the guide now says "laptop graphics card" throughout, and GPU is expanded at first use. Pictures regenerated with the same wording.
3. Score quoted on the 0-to-1 scale only (0.162; changes as +0.001, -0.009, about -0.031; recomputed from the repo's point figures divided by 100). The "16 out of 100" wording is gone.
4. FP16 file larger than FP32 (8.1 versus 5.2 MB): no explanation found in the repo or docs; the guide says so and does not guess. Added to the questions file. FP32 as the starting point is explained, with the note that the FP32-to-TensorRT speed gap is not a fair test of shrinking.
5. Governor: one toy example kept; the 25% margin and 26,700 figures moved here. Upgrade needs a fit at 0.7 x 0.75 x capacity, so the tracks level (20,000 bps at 70%) needs about 26,700 bps to climb back. "Hysteresis" is no longer used; a thermostat comparison explains the memory.
6. CoT paragraph cut to where, which way and how fast.
7. RT-DETR, SSD, AGPL detail and the "100 times load" remark removed from the guide (AGPL and the alternatives stay in earlier notes above).
8. Terms defined inline; the glossary was removed, per the form rule against glossary boxes.
9. "AI coding agents wrote the code" is stated once in the opening and once by reference in the build chapter; the Q&A repeat was dropped. "My target titles" now states the roles from the author context (AI engineer, forward-deployed AI engineer, applied AI/ML engineer). "Questions file" references removed. No tool is named as builder.
10. Testing and results trimmed to headline figures; noise-floor, interleaved-round and spread detail reduced to plain sentences. The "no customer" row of the recruiter table was merged into the text. The numbers in the guide match the saved result files.
