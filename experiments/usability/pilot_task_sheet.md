# VerifAI Pilot Usability Study (Week 18)

## Goal
Evaluate whether a first-time user can run a deepfake prediction and understand the output.

## Setup (facilitator)
- Backend running locally:
  `python -m uvicorn backend.main:app --reload`
- Participant uses Swagger UI:
  http://127.0.0.1:8000/docs
- Facilitator provides 1 test video file path to upload.

## Task
1. Open the provided link: http://127.0.0.1:8000/docs
2. Find POST `/api/predict`
3. Click “Try it out”
4. Upload the provided test video
5. Click “Execute”
6. Interpret the output:
   - `label` (real/fake)
   - `prob_fake` (0–1)

## Questions (qualitative)
1. What did you expect to happen vs what happened?
2. Was anything confusing or unclear?
3. What would you change to make it easier?
4. How confident would you be using this result? Why?

## Notes recorded by facilitator
- Time to complete: 5.4 minutes
- User comments: Task completed successfully; minor confusion around output interpretation
