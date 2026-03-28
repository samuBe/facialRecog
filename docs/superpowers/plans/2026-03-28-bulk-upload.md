# Bulk Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bulk face enrollment — upload multiple images, review detected faces in a modal, enroll all approved faces in a single API call.

**Architecture:** New `POST /enroll/bulk` endpoint accepts an array of entries in one transaction. Client-side processing pipeline iterates through images, runs face-api.js detection, and presents a review modal. On confirm, one request sends all embeddings to the bulk endpoint.

**Tech Stack:** FastAPI + Pydantic (backend), vanilla JS + face-api.js (frontend), SQLite (storage)

**Spec:** `docs/superpowers/specs/2026-03-28-bulk-upload-design.md`

---

### Task 1: Add `POST /enroll/bulk` endpoint

**Files:**
- Modify: `server/main.py:27-38` (add new Pydantic models after `EnrollRequest`)
- Modify: `server/main.py:94-109` (refactor `upsert_identity` to support optional commit)
- Modify: `server/main.py:170-178` (add new endpoint after existing `/enroll`)

- [ ] **Step 1: Add Pydantic models for bulk enroll**

Add after the `EnrollRequest` class (line 38) in `server/main.py`:

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

- [ ] **Step 2: Refactor `upsert_identity` to accept `commit` parameter**

Change the function signature at line 94 to accept `commit=True`, so bulk can call it without per-entry commits:

```python
def upsert_identity(conn: sqlite3.Connection, label: str, embedding: list[float], metadata: dict[str, Any] | None = None, *, commit: bool = True) -> None:
    unit_vec = embedding_to_unit(embedding)
    conn.execute(
        """
        INSERT INTO identities(label, embedding, metadata)
        VALUES (?, ?, ?)
        ON CONFLICT(label)
        DO UPDATE SET embedding=excluded.embedding, metadata=excluded.metadata
        """,
        (
            label,
            json.dumps(unit_vec.tolist()),
            json.dumps(metadata) if metadata else None,
        ),
    )
    if commit:
        conn.commit()
```

- [ ] **Step 3: Add the `/enroll/bulk` endpoint**

Add after the existing `/enroll` endpoint (after line 178):

```python
@app.post("/enroll/bulk", response_model=BulkEnrollResponse)
def enroll_bulk(payload: BulkEnrollRequest) -> BulkEnrollResponse:
    results: list[EntryResult] = []
    enrolled = 0
    errors = 0

    with get_connection() as conn:
        for entry in payload.entries:
            try:
                upsert_identity(conn, entry.label, entry.embedding, entry.metadata, commit=False)
                results.append(EntryResult(label=entry.label, status="saved"))
                enrolled += 1
            except Exception as exc:
                results.append(EntryResult(label=entry.label, status="error", detail=str(exc)))
                errors += 1
        conn.commit()

    return BulkEnrollResponse(enrolled=enrolled, errors=errors, results=results)
```

- [ ] **Step 4: Verify the server reloads without errors**

Run: check the uvicorn output or `curl -s http://localhost:8000/health`
Expected: `{"status":"ok"}`

- [ ] **Step 5: Test the endpoint manually**

Run:
```bash
curl -s -X POST http://localhost:8000/enroll/bulk \
  -H "Content-Type: application/json" \
  -d '{"entries":[{"label":"test_bulk_1","embedding":['"$(python3 -c "print(','.join(['0.1']*128))")"']},{"label":"test_bulk_2","embedding":['"$(python3 -c "print(','.join(['0.2']*128))")"']}]}' | python3 -m json.tool
```
Expected: `{"enrolled": 2, "errors": 0, "results": [...]}`

- [ ] **Step 6: Commit**

```bash
git add server/main.py
git commit -m "feat: add POST /enroll/bulk endpoint for batch face enrollment"
```

---

### Task 2: Add Bulk Upload button and hidden file input to HTML

**Files:**
- Modify: `client/index.html:43-49` (add button in Database Ops section)

- [ ] **Step 1: Add the bulk upload button and hidden multi-file input**

Add after the `<div class="separator"></div>` / `<h2>Database Ops</h2>` block (after line 45), before the Agent Label input:

```html
          <label class="button">
            Bulk Upload
            <input id="bulkUploadInput" type="file" accept="image/*" multiple hidden />
          </label>
```

- [ ] **Step 2: Verify it renders**

Open `http://localhost:8000` and confirm "Bulk Upload" button appears in the sidebar under "Database Ops".

- [ ] **Step 3: Commit**

```bash
git add client/index.html
git commit -m "feat: add bulk upload button to sidebar"
```

---

### Task 3: Add review modal HTML and CSS

**Files:**
- Modify: `client/index.html:69-70` (add modal markup before closing `</main>`)
- Modify: `client/styles.css:277` (add modal styles before the media query)

- [ ] **Step 1: Add modal HTML**

Add before the closing `</main>` tag (before line 70 in `index.html`):

```html
      <div id="bulkModal" class="bulk-modal" hidden>
        <div class="bulk-modal-content">
          <h2>Bulk Upload Review</h2>
          <p id="bulkProgress" class="bulk-progress"></p>
          <div id="bulkReviewList" class="bulk-review-list"></div>
          <div class="bulk-modal-actions">
            <button id="bulkEnrollBtn" disabled>Enroll All</button>
            <button id="bulkCancelBtn">Cancel</button>
          </div>
        </div>
      </div>
```

- [ ] **Step 2: Add modal CSS**

Add before the `@media` query (before line 278 in `styles.css`):

```css
.bulk-modal {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(1, 5, 8, 0.88);
  backdrop-filter: blur(6px);
}

.bulk-modal[hidden] {
  display: none;
}

.bulk-modal-content {
  width: min(800px, 90vw);
  max-height: 80vh;
  overflow-y: auto;
  border: 1px solid var(--line-strong);
  background: linear-gradient(160deg, rgba(3, 18, 27, 0.95), rgba(1, 11, 18, 0.95));
  padding: 20px;
  box-shadow: 0 0 60px rgba(0, 210, 255, 0.2);
}

.bulk-modal-content h2 {
  margin-bottom: 14px;
}

.bulk-progress {
  color: var(--text-dim);
  margin: 0 0 14px;
}

.bulk-review-list {
  display: grid;
  gap: 10px;
  margin-bottom: 16px;
}

.bulk-review-item {
  display: flex;
  align-items: center;
  gap: 12px;
  border: 1px solid var(--line);
  background: rgba(4, 18, 27, 0.8);
  padding: 10px;
}

.bulk-review-item.no-face {
  opacity: 0.45;
}

.bulk-review-item.duplicate {
  border-color: rgba(255, 200, 50, 0.6);
}

.bulk-review-label {
  flex: 1;
  font-family: "Orbitron", sans-serif;
  font-size: 0.85rem;
  text-transform: uppercase;
}

.bulk-review-status {
  font-size: 0.75rem;
  color: var(--text-dim);
}

.bulk-review-faces {
  display: flex;
  gap: 6px;
}

.bulk-face-option {
  border: 1px solid var(--line);
  padding: 2px;
  cursor: pointer;
  background: transparent;
}

.bulk-face-option.selected {
  border-color: var(--line-strong);
  box-shadow: 0 0 10px rgba(52, 245, 255, 0.4);
}

.bulk-face-option canvas {
  width: 56px;
  height: 56px;
  display: block;
}

.bulk-modal-actions {
  display: flex;
  gap: 10px;
}

.bulk-modal-actions button {
  flex: 1;
  border: 1px solid var(--line);
  background: rgba(7, 26, 37, 0.85);
  color: var(--text);
  font-family: inherit;
  padding: 10px 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  cursor: pointer;
}

.bulk-modal-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

- [ ] **Step 3: Verify modal is hidden by default**

Open `http://localhost:8000`, confirm no visual change (modal has `hidden` attribute).

- [ ] **Step 4: Commit**

```bash
git add client/index.html client/styles.css
git commit -m "feat: add bulk upload review modal markup and styles"
```

---

### Task 4: Implement bulk processing pipeline and modal logic in JS

**Files:**
- Modify: `client/app.js:5-20` (add DOM references for new elements)
- Modify: `client/app.js:356` (add event listeners at end of file)

This is the core task. Add all bulk upload logic to `app.js`.

- [ ] **Step 1: Add DOM references**

Add after the existing DOM references (after line 20, before `const video`):

```javascript
const bulkUploadInput = document.getElementById("bulkUploadInput");
const bulkModal = document.getElementById("bulkModal");
const bulkProgress = document.getElementById("bulkProgress");
const bulkReviewList = document.getElementById("bulkReviewList");
const bulkEnrollBtn = document.getElementById("bulkEnrollBtn");
const bulkCancelBtn = document.getElementById("bulkCancelBtn");

let bulkResults = [];
```

- [ ] **Step 2: Add helper to derive label from filename**

Add after the `bulkResults` variable:

```javascript
function labelFromFilename(filename) {
  const name = filename.replace(/\.[^/.]+$/, "").trim();
  return name.slice(0, 80) || "unnamed";
}
```

- [ ] **Step 3: Add function to load an image from a File as a promise**

```javascript
function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Failed to load ${file.name}`));
    };
    img.src = url;
  });
}
```

- [ ] **Step 4: Add function to detect faces on an offscreen canvas**

```javascript
async function detectFacesInImage(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext("2d").drawImage(img, 0, 0);

  const detections = await faceapi
    .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }))
    .withFaceLandmarks()
    .withFaceDescriptors();

  return { canvas, detections };
}
```

- [ ] **Step 5: Add function to create a face thumbnail canvas**

```javascript
function makeFaceThumb(sourceCanvas, box, size = 56) {
  const thumb = document.createElement("canvas");
  thumb.width = size;
  thumb.height = size;
  const pad = 24;
  const sx = Math.max(0, box.x - pad);
  const sy = Math.max(0, box.y - pad);
  const sw = Math.min(sourceCanvas.width - sx, box.width + pad * 2);
  const sh = Math.min(sourceCanvas.height - sy, box.height + pad * 2);
  thumb.getContext("2d").drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, size, size);
  return thumb;
}
```

- [ ] **Step 6: Add the main bulk processing function**

This function processes all selected files, populates `bulkResults`, and renders the review modal:

```javascript
async function processBulkUpload(files) {
  const fileList = Array.from(files).slice(0, 50);
  bulkResults = [];
  bulkReviewList.innerHTML = "";
  bulkModal.hidden = false;
  bulkEnrollBtn.disabled = true;

  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    const label = labelFromFilename(file.name);
    bulkProgress.textContent = `Processing ${i + 1}/${fileList.length}: ${file.name}`;

    try {
      const img = await loadImageFile(file);
      const { canvas, detections } = await detectFacesInImage(img);

      if (detections.length === 0) {
        bulkResults.push({ label, faces: [], selectedIndex: -1, status: "no-face" });
      } else {
        const largestIndex = detections.reduce((best, d, idx, arr) =>
          d.detection.box.area > arr[best].detection.box.area ? idx : best, 0);
        bulkResults.push({
          label,
          faces: detections.map((d, idx) => ({
            descriptor: Array.from(d.descriptor),
            thumb: makeFaceThumb(canvas, d.detection.box),
          })),
          selectedIndex: largestIndex,
          status: "ready",
        });
      }
    } catch (err) {
      bulkResults.push({ label, faces: [], selectedIndex: -1, status: "error", error: err.message });
    }
  }

  renderBulkReview();
}
```

- [ ] **Step 7: Add function to render the review list**

```javascript
function renderBulkReview() {
  bulkReviewList.innerHTML = "";

  const labelCounts = {};
  bulkResults.forEach((r) => {
    if (r.status === "ready") {
      labelCounts[r.label] = (labelCounts[r.label] || 0) + 1;
    }
  });

  bulkResults.forEach((result, resultIndex) => {
    const row = document.createElement("div");
    row.className = "bulk-review-item";

    if (result.status === "no-face" || result.status === "error") {
      row.classList.add("no-face");
    }
    if (labelCounts[result.label] > 1 && result.status === "ready") {
      row.classList.add("duplicate");
    }

    const labelEl = document.createElement("span");
    labelEl.className = "bulk-review-label";
    labelEl.textContent = result.label;

    const statusEl = document.createElement("span");
    statusEl.className = "bulk-review-status";

    if (result.status === "no-face") {
      statusEl.textContent = "No face detected";
    } else if (result.status === "error") {
      statusEl.textContent = result.error || "Failed to load";
    } else if (labelCounts[result.label] > 1) {
      statusEl.textContent = "Duplicate label";
    } else {
      statusEl.textContent = `${result.faces.length} face(s)`;
    }

    row.appendChild(labelEl);

    if (result.status === "ready" && result.faces.length > 0) {
      const facesContainer = document.createElement("div");
      facesContainer.className = "bulk-review-faces";

      result.faces.forEach((face, faceIndex) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "bulk-face-option";
        if (faceIndex === result.selectedIndex) btn.classList.add("selected");
        btn.appendChild(face.thumb);
        btn.addEventListener("click", () => {
          bulkResults[resultIndex].selectedIndex = faceIndex;
          renderBulkReview();
        });
        facesContainer.appendChild(btn);
      });

      row.appendChild(facesContainer);
    }

    row.appendChild(statusEl);
    bulkReviewList.appendChild(row);
  });

  bulkProgress.textContent = `${bulkResults.filter((r) => r.status === "ready").length} of ${bulkResults.length} images ready for enrollment.`;
  bulkEnrollBtn.disabled = !bulkResults.some((r) => r.status === "ready" && r.selectedIndex >= 0);
}
```

- [ ] **Step 8: Add the bulk enroll function**

```javascript
async function bulkEnroll() {
  const entries = bulkResults
    .filter((r) => r.status === "ready" && r.selectedIndex >= 0)
    .map((r) => ({
      label: r.label,
      embedding: r.faces[r.selectedIndex].descriptor,
      metadata: { source: "bulk-upload", captured_at: new Date().toISOString() },
    }));

  if (!entries.length) {
    log("No faces to enroll.");
    return;
  }

  bulkEnrollBtn.disabled = true;
  bulkProgress.textContent = `Enrolling ${entries.length} face(s)...`;

  try {
    const response = await fetch(`${API_URL}/enroll/bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries }),
    });

    if (!response.ok) {
      throw new Error(`Bulk enroll failed: ${response.status}`);
    }

    const data = await response.json();
    log(`Bulk enroll complete: ${data.enrolled} saved, ${data.errors} errors.`);

    data.results.forEach((r) => {
      if (r.status === "error") {
        log(`  Error enrolling ${r.label}: ${r.detail}`);
      }
    });

    bulkModal.hidden = true;
  } catch (err) {
    log(`Bulk enroll error: ${err.message}`);
    bulkEnrollBtn.disabled = false;
  }
}
```

- [ ] **Step 9: Add event listeners**

Add at the end of the file, before `window.addEventListener("beforeunload", stopCamera);`:

```javascript
bulkUploadInput.addEventListener("change", (event) => {
  const files = event.target.files;
  if (files.length) {
    processBulkUpload(files);
  }
  bulkUploadInput.value = "";
});

bulkEnrollBtn.addEventListener("click", bulkEnroll);
bulkCancelBtn.addEventListener("click", () => {
  bulkModal.hidden = true;
  bulkResults = [];
});
```

- [ ] **Step 10: Verify end-to-end in the browser**

Open `http://localhost:8000`:
1. Click "Bulk Upload" and select 2-3 images
2. Confirm the review modal appears with detected faces
3. Click "Enroll All" and verify the telemetry log shows results
4. Check enrolled identities via `curl http://localhost:8000/identities`

- [ ] **Step 11: Commit**

```bash
git add client/app.js
git commit -m "feat: implement bulk upload processing pipeline and review modal logic"
```
