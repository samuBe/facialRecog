const MODEL_URLS = [
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/",
  "https://justadudewhohacks.github.io/face-api.js/models/",
];
const API_URL = "http://localhost:8000";

const fheToggleBtn = document.getElementById("fheToggleBtn");
const fheDot = document.getElementById("fheDot");
const fheLabel = document.getElementById("fheLabel");
let fheMode = false;

function setFheStatus(on) {
  fheMode = on;
  fheDot.classList.toggle("active", on);
  fheToggleBtn.classList.toggle("active", on);
  fheLabel.textContent = on ? "FHE ON" : "FHE Mode";
}

async function toggleFhe() {
  if (!fheMode) {
    fheLabel.textContent = "Checking...";
    try {
      const r = await fetch(`${API_URL}/health`);
      const data = await r.json();
      if (data.fhe !== "ok") {
        setFheStatus(false);
        log("FHE backend not available — server reports fhe: " + data.fhe);
        return;
      }
    } catch (err) {
      setFheStatus(false);
      log("Cannot reach server — FHE unavailable: " + err.message);
      return;
    }
    setFheStatus(true);
    log("FHE mode ON — using encrypted backend");
  } else {
    setFheStatus(false);
    log("FHE mode OFF — using plaintext backend");
  }
}

window.__toggleFhe = toggleFhe;
fheToggleBtn.addEventListener("click", toggleFhe);

function apiPath(path) {
  return fheMode ? `${API_URL}/fhe${path}` : `${API_URL}${path}`;
}

const uploadInput = document.getElementById("uploadInput");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const detectBtn = document.getElementById("detectBtn");
const lookupBtn = document.getElementById("lookupBtn");
const enrollBtn = document.getElementById("enrollBtn");
const agentLabelInput = document.getElementById("agentLabel");
const modelDot = document.getElementById("modelDot");
const modelStatus = document.getElementById("modelStatus");
const telemetryLog = document.getElementById("telemetryLog");
const resultsNode = document.getElementById("results");
const noSignal = document.getElementById("noSignal");
const facesStrip = document.getElementById("facesStrip");

const bulkUploadInput = document.getElementById("bulkUploadInput");
const bulkModal = document.getElementById("bulkModal");
const bulkProgress = document.getElementById("bulkProgress");
const bulkReviewList = document.getElementById("bulkReviewList");
const bulkEnrollBtn = document.getElementById("bulkEnrollBtn");
const bulkCancelBtn = document.getElementById("bulkCancelBtn");

let bulkResults = [];
let bulkProcessing = false;

const video = document.getElementById("camera");
const sceneCanvas = document.getElementById("sceneCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const sceneCtx = sceneCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

let cameraStream = null;
let faceDetections = [];
let selectedFaceIndex = -1;
let sourceLoaded = false;

function log(line) {
  const stamp = new Date().toLocaleTimeString();
  telemetryLog.textContent = `[${stamp}] ${line}\n${telemetryLog.textContent}`;
}

function setModelReady(ready, message) {
  modelStatus.textContent = message;
  modelDot.classList.toggle("ready", ready);
}

function clearResults() {
  resultsNode.innerHTML = "";
}

function resizeCanvases(width, height) {
  sceneCanvas.width = width;
  sceneCanvas.height = height;
  overlayCanvas.width = width;
  overlayCanvas.height = height;
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  faceDetections.forEach((face, index) => {
    const { x, y, width, height } = face.detection.box;
    overlayCtx.strokeStyle = index === selectedFaceIndex ? "#34f5ff" : "rgba(52, 245, 255, 0.45)";
    overlayCtx.lineWidth = index === selectedFaceIndex ? 3 : 1.4;
    overlayCtx.strokeRect(x, y, width, height);

    overlayCtx.fillStyle = "rgba(1, 8, 14, 0.8)";
    overlayCtx.fillRect(x, Math.max(0, y - 20), 90, 18);
    overlayCtx.fillStyle = "#34f5ff";
    overlayCtx.font = "12px Share Tech Mono";
    overlayCtx.fillText(`FACE ${index + 1}`, x + 6, Math.max(12, y - 8));
  });
}

function setSelection(index) {
  selectedFaceIndex = index;
  lookupBtn.disabled = index < 0;
  enrollBtn.disabled = index < 0;
  drawOverlay();

  [...facesStrip.querySelectorAll(".face-thumb")].forEach((node, nodeIndex) => {
    node.classList.toggle("active", nodeIndex === index);
  });

  if (index >= 0) {
    log(`Selected face ${index + 1} for embedding lookup.`);
  }
}

function renderThumbnails() {
  facesStrip.innerHTML = "";
  faceDetections.forEach((face, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "face-thumb";

    const thumb = document.createElement("canvas");
    const tctx = thumb.getContext("2d");
    const { x, y, width, height } = face.detection.box;
    const pad = 24;
    const sx = Math.max(0, x - pad);
    const sy = Math.max(0, y - pad);
    const sw = Math.min(sceneCanvas.width - sx, width + pad * 2);
    const sh = Math.min(sceneCanvas.height - sy, height + pad * 2);

    thumb.width = 96;
    thumb.height = 96;
    tctx.drawImage(sceneCanvas, sx, sy, sw, sh, 0, 0, thumb.width, thumb.height);

    const caption = document.createElement("p");
    caption.textContent = `Face ${index + 1}`;

    card.appendChild(thumb);
    card.appendChild(caption);
    card.addEventListener("click", () => setSelection(index));

    facesStrip.appendChild(card);
  });
}

async function loadModels() {
  for (const url of MODEL_URLS) {
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(url),
        faceapi.nets.faceLandmark68Net.loadFromUri(url),
        faceapi.nets.faceRecognitionNet.loadFromUri(url),
      ]);

      setModelReady(true, "Models loaded");
      detectBtn.disabled = false;
      log(`face-api models loaded from ${url}`);

      return;
    } catch (error) {
      log(`Model source failed (${url}): ${error.message}`);
    }
  }

  setModelReady(false, "Model load failed");
}

function drawImageToScene(imageLike, width, height) {
  resizeCanvases(width, height);
  sceneCtx.clearRect(0, 0, width, height);
  sceneCtx.drawImage(imageLike, 0, 0, width, height);
  overlayCtx.clearRect(0, 0, width, height);

  sourceLoaded = true;
  noSignal.style.display = "none";
  clearResults();
  facesStrip.innerHTML = "";
  faceDetections = [];
  setSelection(-1);
}

async function handleUpload(file) {
  const image = new Image();
  image.onload = () => {
    drawImageToScene(image, image.naturalWidth, image.naturalHeight);
    log(`Image loaded (${image.naturalWidth}x${image.naturalHeight}). Ready to extract faces.`);
  };
  image.src = URL.createObjectURL(file);
}

async function startCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
  video.srcObject = cameraStream;
  await video.play();

  captureBtn.disabled = false;
  stopCameraBtn.disabled = false;
  log("Camera stream active.");
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    video.srcObject = null;
    captureBtn.disabled = true;
    stopCameraBtn.disabled = true;
    log("Camera stopped.");
  }
}

async function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) {
    log("No camera frame available.");
    return;
  }

  drawImageToScene(video, video.videoWidth, video.videoHeight);
  log("Frame captured to analysis stage.");
}

async function detectFaces() {
  if (!sourceLoaded) {
    log("Load an image first.");
    return;
  }

  const detections = await faceapi
    .detectAllFaces(sceneCanvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }))
    .withFaceLandmarks()
    .withFaceDescriptors();

  faceDetections = detections;
  if (!detections.length) {
    setSelection(-1);
    drawOverlay();
    facesStrip.innerHTML = "";
    log("No faces detected.");
    return;
  }

  renderThumbnails();
  setSelection(0);
  log(`Extracted ${detections.length} face(s). Select one to continue.`);
}

function selectedDescriptor() {
  if (selectedFaceIndex < 0 || selectedFaceIndex >= faceDetections.length) {
    return null;
  }
  return Array.from(faceDetections[selectedFaceIndex].descriptor);
}

async function lookupFace() {
  const embedding = selectedDescriptor();
  if (!embedding) {
    log("Select a face before lookup.");
    return;
  }

  try {
    const response = await fetch(`${apiPath("/search")}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding, top_k: 5, threshold: 0.35 }),
    });

    if (!response.ok) {
      throw new Error(`Lookup failed: ${response.status}`);
    }

    const data = await response.json();
    renderMatches(data.matches || []);
    log(`Lookup complete. ${data.count} candidate(s) above threshold.`);
  } catch (error) {
    log(error.message);
  }
}

function renderMatches(matches) {
  resultsNode.innerHTML = "";

  if (!matches.length) {
    const empty = document.createElement("div");
    empty.className = "result-card warning";
    empty.innerHTML = "<h3>No match</h3><p>No embedding passed threshold.</p>";
    resultsNode.appendChild(empty);
    return;
  }

  matches.forEach((match, index) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const meta = match.metadata ? Object.entries(match.metadata).map(([k, v]) => `${k}: ${v}`).join(" | ") : "no metadata";

    card.innerHTML = `
      <h3>#${index + 1} ${match.label}</h3>
      <p>Similarity: ${(match.similarity * 100).toFixed(2)}%</p>
      <p>${meta}</p>
    `;
    resultsNode.appendChild(card);
  });
}

async function enrollFace() {
  const embedding = selectedDescriptor();
  if (!embedding) {
    log("Select a face before enrollment.");
    return;
  }

  const label = agentLabelInput.value.trim();
  if (!label) {
    log("Provide an agent label before enrollment.");
    return;
  }

  try {
    const response = await fetch(`${apiPath("/enroll")}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label,
        embedding,
        metadata: {
          source: "browser-client",
          captured_at: new Date().toISOString(),
        },
      }),
    });

    if (!response.ok) {
      const errorBody = await response.json();
      throw new Error(errorBody.detail || `Enroll failed: ${response.status}`);
    }

    log(`Enrollment updated for ${label}.`);
  } catch (error) {
    log(error.message);
  }
}

function labelFromFilename(filename) {
  const name = filename.replace(/\.[^/.]+$/, "").trim();
  return name.slice(0, 80) || "unnamed";
}

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

async function processBulkUpload(files) {
  if (bulkProcessing) return;
  bulkProcessing = true;
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

  bulkProcessing = false;
  renderBulkReview();
}

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

    const facesContainer = document.createElement("div");
    facesContainer.className = "bulk-review-faces";

    if (result.status === "ready" && result.faces.length > 0) {
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
    }

    const infoEl = document.createElement("div");
    infoEl.className = "bulk-review-info";

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
      statusEl.textContent = "Duplicate label — last wins";
    } else {
      statusEl.textContent = `${result.faces.length} face(s) detected`;
    }

    infoEl.appendChild(labelEl);
    infoEl.appendChild(statusEl);

    row.appendChild(facesContainer);
    row.appendChild(infoEl);
    bulkReviewList.appendChild(row);
  });

  bulkProgress.textContent = `${bulkResults.filter((r) => r.status === "ready").length} of ${bulkResults.length} images ready for enrollment.`;
  bulkEnrollBtn.disabled = !bulkResults.some((r) => r.status === "ready" && r.selectedIndex >= 0);
}

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
    const response = await fetch(`${apiPath("/enroll/bulk")}`, {
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
    bulkResults = [];
  } catch (err) {
    log(`Bulk enroll error: ${err.message}`);
    bulkEnrollBtn.disabled = false;
  }
}

overlayCanvas.addEventListener("click", (event) => {
  if (!faceDetections.length) {
    return;
  }

  const rect = overlayCanvas.getBoundingClientRect();
  const scaleX = overlayCanvas.width / rect.width;
  const scaleY = overlayCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  let picked = -1;
  faceDetections.forEach((face, index) => {
    const box = face.detection.box;
    if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
      picked = index;
    }
  });

  if (picked >= 0) {
    setSelection(picked);
  }
});

uploadInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    handleUpload(file);
  }
});

startCameraBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (error) {
    log(`Cannot start camera: ${error.message}`);
  }
});

captureBtn.addEventListener("click", captureFrame);
stopCameraBtn.addEventListener("click", stopCamera);
detectBtn.addEventListener("click", detectFaces);
lookupBtn.addEventListener("click", lookupFace);
enrollBtn.addEventListener("click", enrollFace);

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

window.addEventListener("beforeunload", stopCamera);
loadModels();
