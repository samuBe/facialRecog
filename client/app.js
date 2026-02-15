const MODEL_URLS = [
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/",
  "https://justadudewhohacks.github.io/face-api.js/models/",
];
const API_URL = "http://localhost:8000";

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
    const response = await fetch(`${API_URL}/search`, {
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
    const response = await fetch(`${API_URL}/enroll`, {
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

window.addEventListener("beforeunload", stopCamera);
loadModels();
