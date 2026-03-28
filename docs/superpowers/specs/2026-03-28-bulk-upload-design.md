# Bulk Upload Design

## Summary

Add bulk face enrollment: upload multiple images, review detected faces, and enroll them in a single request. Filenames (minus extension) become labels.

## User Flow

1. User clicks "Bulk Upload" in the Database Ops sidebar section
2. File picker opens with multi-select enabled (accepts image files, max 50 files)
3. Client processes each image sequentially through face-api.js, extracting faces and embeddings. A progress indicator shows status (e.g., "Processing 3/10...").
4. A review modal overlay displays results grouped by image:
   - **Single face**: thumbnail + filename label, pre-selected for enrollment
   - **Multiple faces**: all detected faces shown, largest face pre-selected. User can click a different face to change selection.
   - **No faces**: grayed out with "No face detected", excluded from enrollment
   - **Failed to load**: grayed out with error message, excluded
5. If two files produce the same label, the modal flags the duplicate. Last occurrence wins if both are enrolled.
6. User confirms enrollment; client sends all approved faces to `POST /enroll/bulk` in one request
7. Existing labels are overwritten (upsert, matching current `/enroll` behavior)
8. Modal closes. Results (including any per-entry errors) summarized in the telemetry log.

## Label Derivation

- File extension is stripped: `agent_vega.jpg` becomes `agent_vega`
- Leading/trailing whitespace is trimmed
- Labels exceeding 80 characters are truncated to 80
- No other character transformations (spaces, parentheses, etc. are preserved)

## Backend Changes

### New endpoint: `POST /enroll/bulk`

Accepts an array of enrollment entries processed in a single database transaction.

**Pydantic models:**

```python
class BulkEnrollRequest(BaseModel):
    entries: list[EnrollRequest] = Field(..., min_length=1, max_length=50)

class EntryResult(BaseModel):
    label: str
    status: str  # "saved" or "error"
    detail: str | None = None

class BulkEnrollResponse(BaseModel):
    enrolled: int
    errors: int
    results: list[EntryResult]
```

**Request body:**
```json
{
  "entries": [
    {
      "label": "agent_vega",
      "embedding": [0.1, 0.2, ...],
      "metadata": {"source": "bulk-upload", "captured_at": "..."}
    }
  ]
}
```

**Response:**
```json
{
  "enrolled": 4,
  "errors": 1,
  "results": [
    {"label": "agent_vega", "status": "saved", "detail": null},
    {"label": "bad_entry", "status": "error", "detail": "Embedding norm cannot be zero"}
  ]
}
```

**Behavior:**
- Validates all entries first using existing `EnrollRequest` validation
- Opens a single SQLite transaction for all writes
- Calls the upsert SQL directly (not `upsert_identity`, which auto-commits) to avoid per-entry commits
- Entries that fail during write are recorded as errors; the transaction commits all successful writes
- Maximum 50 entries per request (enforced by Pydantic `max_length`)

The existing `POST /enroll` endpoint remains unchanged.

## Frontend Changes

### Bulk Upload Button
- Added to the "Database Ops" section in the sidebar, below the existing Enroll/Lookup buttons
- Hidden file input with `multiple` attribute, accepts image types

### Processing Pipeline
- Iterates through selected files sequentially (intentional simplicity tradeoff; sequential processing avoids complexity and is acceptable for up to 50 images)
- For each file: load as Image, run face-api detection, collect results
- Progress indicator shown during processing

### Review Modal
- Overlay modal that appears after all images are processed
- Each row shows: filename, thumbnail(s) of detected faces, derived label, status
- For multi-face images: clickable face thumbnails; largest face pre-selected
- For no-face images: grayed out, excluded
- "Enroll All" button at the bottom to confirm; "Cancel" to abort
- After enrollment: modal closes, telemetry log shows summary including any per-entry errors

### Error Handling
- Images that fail to load: flagged in review as "Failed to load"
- Network error on bulk enroll: error shown in telemetry log with details
- Per-entry validation errors from the server: shown in post-enrollment summary in telemetry log

## What Does Not Change

- Single-image upload and enrollment flow
- Face detection logic (face-api.js, client-side)
- Existing `POST /enroll` and `POST /search` endpoints
- Database schema
