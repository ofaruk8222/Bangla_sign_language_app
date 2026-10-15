# PROJECT_CONTEXT.md
> Full continuity document for the Real-Time Bangla Sign Language (BdSL) Translator thesis project.
> Feed this file to any AI assistant to resume work exactly where we left off.

## 1. PROJECT OVERVIEW
- Goal: Real-time, offline-capable mobile app (Flutter) that translates Bangla Sign Language from live camera into text (subtitle-style).
- Team: 3 students; timeline 6–7 months; final-year CSE thesis.
- ML approach: MediaPipe Holistic keypoint extraction → sequence model (Bi-LSTM baseline) → word classification → (future) sentence assembly + Flutter/ONNX deployment.
- Status (as of last message): Pipeline verified end-to-end on BdSLW401 (10-class test run). User paused to ask a batch of conceptual questions (pending), then requested this context file.

## 2. HARDWARE / ENVIRONMENT (CRITICAL PITFALLS)
- User laptop: Windows 11, CPU-only (no CUDA). Teammate PC: Windows 11, Intel, Python 3.11.6.
- Python MUST be 3.11.x 64-bit. Python 3.14 broke MediaPipe (pip installed a stub; both `mp.solutions` and `mp.Holistic` missing).
- Virtualenv: `venv311` in project folder. Activate: `.\venv311\Scripts\Activate.ps1`
  - If "running scripts is disabled": `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Install commands (MUST be split; a single command with --index-url makes pip search for mediapipe on the PyTorch index → "no matching distribution"):
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install mediapipe==0.10.14 opencv-python tqdm numpy
  ```
- MediaPipe 0.10.14 API syntax (the only one that works there):
  ```python
  with mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  ```
  (`mp.Holistic(...)` raises AttributeError on 0.10.14; `mp.solutions...` was missing only on the broken 3.14 install.)
- Other past fixes: WinError 32 on cv2.pyd → close IDE + kill python processes; Mendeley HTTP 403 → use Kaggle mirror; "num_samples=0" ValueError → stale empty .npy files, delete processed_data/*.npy; IDE "Select environment manager" popup → ignore, use terminal.
- Google Colab abandoned (caching bugs, cannot pin Python version). All work is local.

## 3. PATHS & FILES
- User laptop project folder: `D:\4-I\Projects\Thesis\Sample testing with english\sample try by me\`
  - Contains: `venv311\`, `train_pipeline.py` (WLASL/ASL version), `create_subset.py`, `diagnosis.py`, `test_mediapipe.py`, `train_bdsl401.py` (current BdSL script)
- BdSLW401 dataset root: `D:\4-I\Projects\Thesis\BdSLW401\`
  - `Front\Front\{train,test,val}\*.mp4`, `Lateral\Lateral\{train,test,val}\*.mp4`, `bdsl words-complete.pdf`
  - Filename convention: `W###S##F_##.mp4` (W=word id, S=signer, F/L=view, ##=trial). Class index = int(W###)-1.
- Processed outputs: `D:\4-I\Projects\Thesis\processed_bdsl401\{X_train,y_train,X_test,y_test}.npy`
- Teammate PC workspace: `D:\for thesis\` (venv, wlasl_small_subset, train_pipeline.py) — secondary machine, same setup replicated.
- Old ASL data: `D:\4-I\Projects\Thesis\Sample testing with english\ASL Dataset\videos` + `WLASL_v0.3.json`.

## 4. DATASET DECISION LOG (do not re-litigate)
- WLASL (ASL, 2000 glosses): used ONLY to verify pipeline; 10-class subset proven.
- Kaggle "bangla-sign-language-video-dataset" (sumon3455, 9k videos): REJECTED — files are pre-rendered low-quality MediaPipe skeleton .avi visualizations, not raw footage; landmarks cannot be re-extracted.
- KU-BdSL (Mendeley scpvm2nbkm; Kaggle mirror mdshahariarhossain/ku-bdsl-eee385l-dataset): REJECTED for main pipeline — static JPG images, no temporal signal for Bi-LSTM.
- BDSL-49, BDSL-47, OkkhorNama, IsharaLipi: REJECTED — static image datasets (CNN/object-detection oriented).
- BdSLW-11: REJECTED — only 11 classes.
- BdSLW60 (hasaniut/bdslw60): HELD IN RESERVE — 60 words, 9,307 instances, JSON landmarks + video-clip versions; good sanity-check/benchmark.
- BdSLW52 (mdshahriaremon/bdslw52): HELD IN RESERVE — 52 words incl. verbs/grammar for future sentence-level translation.
- UAP professor's 198k-video dataset: permission email sent, no reply yet; follow-up template drafted.
- ✅ BdSLW401 (hasanssl/bdslw401): CHOSEN PRIMARY. 401 words, 102,176 videos, 18 signers (15M/3F, ages 18–25), front+lateral webcam views, raw MP4, official train/test/val splits, ~52 GB. Download took user 2 days (power outages) — completed.

## 5. PIPELINE (train_bdsl401.py) SPEC
1. Per-frame MediaPipe Holistic on raw MP4 (OpenCV read, BGR→RGB).
2. Features/frame = 100 floats: 8 body pts (idx 11,12,13,14,15,16,23,24 → 16 coords, neck-anchored, scaled by head–neck distance) + 21 left-hand pts (wrist-anchored, 42) + 21 right-hand pts (42).
3. Missing-hand frames (zeros) linearly interpolated from neighbors.
4. Sequence resampled/padded to TARGET_FRAMES=60.
5. Labels parsed from filename (W### → class idx). Train/test taken from dataset's own splits (Front view only for now).
6. Saved as .npy; training skips re-extraction if non-empty .npy exists.
7. Model = custom 2-layer Bidirectional LSTM ("Baseline Bi-LSTM"): input 100, hidden 128, bidirectional, dropout 0.2, final-timestep → Linear(num_classes). Thesis phrasing: "Many-to-One Bidirectional LSTM for skeleton-based action recognition, chosen for lightweight offline mobile inference."
8. Training: CrossEntropyLoss, Adam lr=1e-3, BATCH=16, EPOCHS=20, CPU device.
- MAX_CLASSES flag gates how many word ids (W001..W{MAX}) are processed; test run used 10.

## 6. RESULTS SO FAR
- WLASL 10-class: "Pipeline Proven!" (one month ago).
- BdSLW401 10-class (W001–W010, Front view): final epoch Train Acc 26.1% / Test Acc 24.9%, loss 1.7714. Interpreted as valid learning baseline (above 10% chance), no overfitting; expected to improve with all 401 classes + augmentation.

## 7. AGREED NEXT STEPS (in order)
1. Answer user's pending conceptual questions (they paused training to ask "a bunch of questions" — not yet asked).
2. Set MAX_CLASSES=401 and run full overnight extraction+training on Front view.
3. Add keypoint-level augmentation (rotation/scale/translation/temporal stretch/frame-drop/noise) to multiply data & signer robustness.
4. Compare Bi-LSTM vs TCN / lightweight Transformer / ST-GCN once baseline exists (swap only the model class; pipeline stays).
5. Later: BdSLW52 verbs for sentence assembly; export TorchScript/ONNX; Flutter integration; cite BdSLW401 paper in thesis.

## 8. USER PREFERENCES (for the assisting AI)
- Give complete, copy-paste-ready code/commands; no vague "simple fix" hand-waving.
- Windows PowerShell syntax; assume CPU-only.
- User is frustrated by repeated environment errors; diagnose root causes explicitly.
- Keep momentum: verify with small runs (MAX_CLASSES=10) before full runs.
- User requested all context preserved in a single .md file (this file).

## APPENDIX A — BdSLW401 WORD MAP (English meanings; Bangla + romanized columns live in `bdsl words-complete.pdf` next to the dataset)
W001 Father | W002 Relatives | W003 Brother | W004 Sister | W005 Wife | W006 Paternal Uncle | W007 Wife of Paternal Uncle | W008 Grandfather | W009 Grandmother | W010 Responsibility | W011 Husband's Younger Brother | W012 Sister's Husband | W013 Wife | W014 Husband of Paternal Aunt | W015 Paternal Aunt | W016 Husband | W017 Maternal Aunt | W018 Husband of Maternal Aunt | W019 Daughter | W020 Mother | W021 Mother's Brother | W022 Mother's Brother's Wife | W023 Husband's Sister | W024 Family | W025 Son | W026 Mother-in-Law | W027 Father-in-Law | W028 Co-wife | W029 Brother's Wife | W030 Nephew (sister's side) | W031 Niece (sister's side) | W032 Wife's Sister's Husband | W033 Good | W034 Nephew (brother's side) | W035 Niece (brother's side) | W036 Husband's Elder Brother | W037 Mango | W038 Potato | W039 Pineapple | W040 Grape | W041 Apple | W042 Biscuit | W043 Jujube | W044 Cake | W045 Tea | W046 Rice (uncooked) | W047 Sugar | W048 Chips | W049 Chocolate | W050 Lentil | W051 Egg | W052 Milk | W053 Feeder | W054 Fruit | W055 Juice | W056 Carrot | W057 Jaggery | W058 Ice Cream | W059 Spicy | W060 Kalojam (sweet) | W061 Food | W062 Date | W063 Banana | W064 Orange | W065 Lemon | W066 Lychee | W067 Salt | W068 Fish | W069 Meat | W070 Sweet | W071 Chilli | W072 Betel Leaf | W073 Guava | W074 Polao | W075 Flatbread | W076 Vegetable | W077 Betel Nut | W078 Palmyra | W079 Tamarind | W080 Melon | W081 Cooked Rice | W082 Alta | W083 Ring | W084 Attar | W085 Landlord | W086 Belt | W087 Bleed | W088 Blouse | W089 Friend | W090 Burqa | W091 Button | W092 Cap | W093 Shawl | W094 Comb | W095 Spectacles | W096 Bangle | W097 Clip | W098 Cream | W099 Data | W100 Indebted | W101 Adopted | W102 Ribbon | W103 Fotua | W104 T-shirt | W105 Watch | W106 Gloves | W107 Necklace | W108 Recipient | W109 Guru | W110 Half-pant | W111 Twin | W112 Shoe | W113 Kohl | W114 Earring | W115 Lipstick | W116 Lungi | W117 Makeup | W118 Owner | W119 Henna | W120 Guest | W121 Socks | W122 Nail polish | W123 Orna | W124 Guardian | W125 Pajama | W126 Panjabi | W127 Pant | W128 Creditor | W129 Petticoat | W130 Perfume | W131 Pocket | W132 Powder | W133 Lover (m) | W134 Lover (f) | W135 Neighbor | W136 Silver | W137 Sandal | W138 Shampoo | W139 Sari | W140 Shave | W141 Shirt | W142 Sherwani | W143 Disciple | W144 Well-wisher | W145 Colleague | W146 Classmate | W147 Same-age | W148 Relation | W149 Gold | W150 Enemy | W151 Sweater | W152 Tie | W153 Tip | W154 Tenant | W155 Anchal | W156 Hanger | W157 Light | W158 Bag | W159 Pillow | W160 Flute | W161 Bathroom | W162 Bowl | W163 Ornament | W164 Bed | W165 Box | W166 Bracelet | W167 Bulb | W168 Chair | W169 Spoon | W170 Umbrella | W171 Hair band | W172 Coat | W173 Cup | W174 Dhuti | W175 Door | W176 Fan | W177 Hookah | W178 Jacket | W179 Window | W180 Men's undergarment | W181 Jeans | W182 Jersey | W183 Basket | W184 Scissors | W185 Kamiz | W186 Fork | W187 Braid | W188 Hairpin | W189 Slipper | W190 Locket | W191 Muffler | W192 Stool | W193 Mosquito net | W194 Nose-pin | W195 Nighty | W196 Nose ring | W197 Anklet | W198 Turban | W199 Plate | W200 Raincoat | W201 Salwar | W202 Scarf | W203 Semiz | W204 Sunglass | W205 Thread | W206 Table | W207 Telephone | W208 Tie-pin | W209 Tikli | W210 Toothbrush | W211 Toothpaste | W212 T-shirt | W213 Tube light | W214 TV | W215 AC | W216 Apartment | W217 Audio cassette | W218 Mirror | W219 Bucket | W220 Sand | W221 Hut | W222 Basin | W223 Bed sheet | W224 Bodna | W225 Bookshelf | W226 Briefcase | W227 Calling bell | W228 Cassette player | W229 Moon | W230 Strainer | W231 Stove | W232 Building | W233 Wall | W234 Brick | W235 Haphazard | W236 Fridge | W237 Fry pan | W238 Flower vase | W239 Carpet | W240 Gamcha | W241 Iron | W242 Broom | W243 Jug | W244 Katha | W245 Kettle | W246 Room | W247 Well | W248 Hut | W249 Blanket | W250 Lift | W251 Mud house | W252 Floor | W253 Candle | W254 Mug | W255 Tube well | W256 Water tap | W257 Curtain | W258 Pressure cooker | W259 Pond | W260 Radio | W261 Handkerchief | W262 Soap | W263 Arrange | W264 Cement | W265 Stairs | W266 Sofa | W267 Suitcase | W268 Switch | W269 Table lamp | W270 Tin house | W271 Torchlight | W272 Ingredients | W273 Veranda | W274 Washing machine | W275 Ashtray | W276 Blender | W277 Boti | W278 Knife | W279 Chimney | W280 Door bolt | W281 Daa (blade) | W282 Matchstick | W283 Detergent | W284 Rope | W285 Grill | W286 Hanger | W287 Lantern | W288 Blanket | W289 Commode | W290 Saw | W291 Microwave | W292 Door mat | W293 Nail | W294 Ramda | W295 Screw | W296 Tank | W297 Wire | W298 Flower tub | W299 Toilet tissue | W300 Lawn | W301 Tar | W302 Bamboo | W303 Roti roller | W304 Bottle | W305 Jar | W306 Shed | W307 Walking stick | W308 Lime | W309 Cover | W310 Cradle | W311 Drum | W312 Drawer | W313 File | W314 Flask | W315 Foam | W316 Folder | W317 Big bowl | W318 Pillar | W319 Sitting place | W320 Cattle house | W321 Mortar | W322 Cooking pan | W323 Hammer | W324 Broom stick | W325 Shoe rack | W326 Sickle | W327 Straw | W328 Cooking spud | W329 Koyel | W330 Cushion | W331 Lighter | W332 Mat | W333 Ladder | W334 Weeder | W335 Hand fan | W336 Pan | W337 Punching machine | W338 Pin | W339 Plastic | W340 Plug | W341 Prasad | W342 Lamp | W343 Watering | W344 Sewing machine | W345 Shower | W346 Stapler | W347 Needle | W348 Tent | W349 Rack | W350 Tiffin carrier | W351 AIDS | W352 Arthritis | W353 Bandage | W354 Capsule | W355 Treatment | W356 Conjunctivitis | W357 Dengue | W358 Doctor | W359 Bite | W360 Weak | W361 Abscess | W362 Heat rash | W363 Goitre | W364 Homeopathy | W365 Hospital | W366 Cardiac disease | W367 Injection | W368 Germ | W369 Tuberculosis | W370 Jaundice | W371 Fever | W372 Cough | W373 Injury | W374 Worm | W375 Mask | W376 Headache | W377 Malaria | W378 Ointment | W379 Heal | W380 Nurse | W381 Paralyzed | W382 Senseless | W383 Insomnia | W384 Malnutrition | W385 No appetite | W386 Uncomfortable | W387 Sick | W388 Surgery | W389 Medicine | W390 Savlon | W391 Healthy | W392 Contagious | W393 Cold | W394 Stethoscope | W395 Tablet | W396 Thermometer | W3