### Baseline sanity test (API predict)
- Input: data/raw/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4
- Expected: HTTP 200 + JSON response with label + prob_fake
- Result: PASS
- Notes: API returned {"label":"real","prob_fake":0.0}

### Wrong file type (.txt)
- Input: data/raw/fake.txt
- Expected: HTTP 400 with clear validation message (not crash)
- Result: PASS (after fix)
- Notes: Initially returned 500. Added API-level file validation in detect.py. Now returns:
  {"detail":"Invalid file type. Supported extensions: .avi, .mov, .mp4"}

#### Fix implemented
- File: backend/app/api/detect.py
- Change: Added extension + empty-file validation before calling inference_service.
- Impact: Converted invalid uploads from 500 → 400 (robust API behaviour).

### Malformed MP4 (created via PowerShell Out-File)
- Input: data/raw/empty.mp4
- Expected: HTTP 400 with a clear validation message (not a crash)
- Result: PASS (after fix)
- Notes: Initially caused 500 due to downstream video decoding/inference RuntimeError. Added MP4 signature validation (checks for 'ftyp' box in first 512 bytes) so malformed uploads return:
  {"detail":"Invalid MP4 file. The uploaded file does not look like a valid video."}

### Truly empty MP4 (0 bytes)
- Input: data/raw/empty_true.mp4
- Expected: HTTP 400 with "Uploaded file is empty"
- Result: PASS
- Notes: Empty uploads are detected before inference and rejected with a clean client error.

## Fix implemented (Week 18)
- File modified: backend/app/api/detect.py
- Changes:
  1) Extension validation (.mp4/.avi/.mov) → prevents wrong file types reaching inference
  2) Empty/malformed upload validation:
     - Reads first 512 bytes and checks MP4 signature ('ftyp')
     - Rejects invalid MP4 containers with HTTP 400
- Impact:
  - Converted invalid uploads from 500 Internal Server Error → 400 with clear messages
  - Improved API reliability and robustness for internal testing + usability study
