# Bulk Upload Design

## Summary

Add bulk face enrollment: upload multiple images, review detected faces, and enroll them in a single request. Filenames become labels.

## User Flow

1. User clicks "Bulk Upload" in the Database Ops sidebar section
2. File picker opens with multi-select enabled (accepts image files)
3. Client processes each image through face-api.js, extracting faces and embeddings
4. A review modal overlay displays results grouped by image:
   - **Single face**: thumbnail + filename label, ready to enroll
   - **Multiple faces**: all detected faces shown; user picks which one to enroll under that filename
   - **No faces**: flagged as "no face detected", excluded from enrollment
5. User confirms enrollment; client sends all approved faces to `POST /enroll/bulk` in one request
6. Existing labels are overwritten (upsert, matching current `/enroll` behavior)
7. Results summarized in the telemetry log

## Backend Changes

### New endpoint: `POST /enroll/bulk`

Accepts an array of enrollment entries and processes them in a single transaction.

**Request body:**
```json
{
  "entries": [
    {
      "label": "Agent Vega",
      "embedding": [0.1, 0.2, ...],
      "metadata": {"source": "bulk-upload", "captured_at": "..."}
    }
  ]
}
```

**Response:**
```json
{
  "enrolled": 5,
  "results": [
    {"label": "Agent Vega", "status": "saved"},
    {"label": "Agent Nova", "status": "saved"}
  ]
}
```

**Behavior:**
- Wraps all upserts in a single SQLite transaction
- Validates each entry (embedding length, label format) and reports per-entry errors
- Entries that fail validation are skipped; successful entries are still committed
- Reuses existing `upsert_identity` logic

The existing `POST /enroll` endpoint remains unchanged.

## Frontend Changes

### Bulk Upload Button
- Added to the "Database Ops" section in the sidebar, below the existing Enroll/Lookup buttons

### Processing Pipeline
- Iterates through selected files sequentially
- For each file: load as Image, run face-api detection, collect results
- Progress indicator shown during processing (e.g., "Processing 3/10...")

### Review Modal
- Overlay modal that appears after all images are processed
- Each image row shows: filename, thumbnail(s) of detected faces, status
- For multi-face images: clickable face thumbnails to select which face to enroll
- For no-face images: grayed out with "No face detected" label
- "Enroll All" button at the bottom to confirm; "Cancel" to abort
- After enrollment: modal closes, telemetry log shows summary

### Error Handling
- Images that fail to load: flagged in review as "Failed to load"
- Network error on bulk enroll: error shown in telemetry log with details
- Per-entry validation errors from the server: shown in a post-enrollment summary

## What Does Not Change

- Single-image upload and enrollment flow
- Face detection logic (face-api.js, client-side)
- Existing `POST /enroll` and `POST /search` endpoints
- Database schema
