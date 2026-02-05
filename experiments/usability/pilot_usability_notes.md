# Pilot Usability Notes (nU=5)

## Overview
A pilot usability study was conducted with five participants to evaluate whether first-time users could successfully run
a deepfake prediction using the VerifAI API and correctly interpret the output. Participants included two technical 
users (Computer Science / Software Engineering background) and three non-technical users (Business, English Literature, 
and Law).

All users were able to complete the task of uploading a video and obtaining a prediction using the Swagger UI interface.

---

## Participant Breakdown
- Technical users: U1, U2
- Non-technical users: U3, U4, U5

---

## Main Issues Identified

### 1) Interpretation of `prob_fake`
This was the most common issue across all participants.

- **Technical users (U1, U2)**:
  - Understood that `prob_fake` is a probability but were unsure how it should be interpreted in practice.
  - Asked what threshold is used to determine the final label and how borderline cases are handled.

- **Non-technical users (U3, U4, U5)**:
  - Did not understand what `prob_fake` represents.
  - Unsure whether a very small number indicates high confidence or uncertainty.

**Evidence:**
- U1: Unclear whether `prob_fake` represents confidence or raw probability.
- U4 (English Literature): Found numerical output difficult to interpret without plain-language explanation.
- U5: Requested guidance on how reliable the prediction is.

---

### 2) Usability of Swagger UI for Non-Technical Users
While Swagger UI was effective for technical participants, it posed challenges for non-technical users.

- **Technical users** navigated the interface quickly and independently.
- **Non-technical users** required brief explanation of:
  - What Swagger UI is
  - The meaning of “Try it out”
  - How to read JSON output

**Evidence:**
- U3 and U5 found the JSON format confusing.
- U4 found the interface intimidating due to technical language.

---

### 3) Desire for Higher-Level Result Explanation
Users consistently requested a more human-readable explanation of the result.

- Suggested improvements included:
  - Plain English interpretation (e.g. “Very likely real”)
  - Example outputs for reference
  - Explanation of what constitutes a “high” or “low” probability

**Evidence:**
- U2 suggested example outputs.
- U4 suggested replacing technical terms with plain language.
- U5 wanted clearer guidance on result reliability.

---

## Changes Implemented During Week 18

Based on internal testing and early usability feedback, the following improvements were implemented:

- Added robust API-level validation:
  - Invalid file types now return HTTP 400 with a clear error message.
  - Malformed or empty MP4 uploads are detected early and rejected gracefully.
- Improved system robustness:
  - Prevented 500 Internal Server Errors caused by malformed inputs.
- Documented decision threshold behavior as part of parameter tuning:
  - Supports clearer interpretation of how labels are derived from probabilities.

Although these changes are backend-focused, they directly improve user experience by reducing confusion and unexpected 
failures.

---

## Future Improvements (Informed by Pilot)

The pilot study highlighted several clear next steps:

1) **Improve output interpretability**
   - Add fields such as `threshold_used` and `confidence_text` to the API response.
   - Provide short explanations for `prob_fake` values.

2) **Provide a user-friendly interface**
   - Develop a simple frontend upload page for non-technical users.
   - Hide raw JSON behind clearer visual explanations.

3) **Documentation enhancements**
   - Include example inputs and outputs in the README.
   - Add a brief “How to interpret results” section.

---

## Conclusion
The pilot usability study demonstrates that VerifAI is functionally usable by first-time users, particularly those with 
technical backgrounds. However, it also reveals clear usability gaps for non-technical users, primarily related to 
result interpretation and interface complexity. The feedback collected directly informs both immediate backend 
improvements and future frontend and documentation development, satisfying the goals of a pilot usability evaluation.
