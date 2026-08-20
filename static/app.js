"use strict";


/* ==========================================================
   ELEMENTS
   ========================================================== */

const examSelect =
    document.getElementById(
        "exam"
    );


const openCameraButton =
    document.getElementById(
        "openCameraButton"
    );

const torchButton =
    document.getElementById(
        "torchButton"
    );

let torchEnabled = false;


const imageUpload =
    document.getElementById(
        "imageUpload"
    );


const cameraContainer =
    document.getElementById(
        "cameraContainer"
    );


const camera =
    document.getElementById(
        "camera"
    );


const captureCanvas =
    document.getElementById(
        "captureCanvas"
    );


const capturedPreview =
    document.getElementById(
        "capturedPreview"
    );


const captureButton =
    document.getElementById(
        "captureButton"
    );


const retakeButton =
    document.getElementById(
        "retakeButton"
    );


const scanButton =
    document.getElementById(
        "scanButton"
    );


const cornerDetectionStatus =
    document.getElementById(
        "cornerDetectionStatus"
    );


const documentBoundaryOverlay =
    document.getElementById(
        "documentBoundaryOverlay"
    );


const loading =
    document.getElementById(
        "loading"
    );


const errorBox =
    document.getElementById(
        "error"
    );


const resultSection =
    document.getElementById(
        "resultSection"
    );


const resultExam =
    document.getElementById(
        "resultExam"
    );


const paperCode =
    document.getElementById(
        "paperCode"
    );


const score =
    document.getElementById(
        "score"
    );


const correct =
    document.getElementById(
        "correct"
    );


const wrong =
    document.getElementById(
        "wrong"
    );


const blank =
    document.getElementById(
        "blank"
    );


const multiple =
    document.getElementById(
        "multiple"
    );


const uncertain =
    document.getElementById(
        "uncertain"
    );


const quality =
    document.getElementById(
        "quality"
    );


const message =
    document.getElementById(
        "message"
    );

const bubbleAnalysisCard =
    document.getElementById(
        "bubbleAnalysisCard"
    );

const bubbleDebugPreview =
    document.getElementById(
        "bubbleDebugPreview"
    );

const resultStream =
    document.getElementById(
        "resultStream"
    );

const kcetStreamSection =
    document.getElementById(
        "kcetStreamSection"
    );

const streamPcmbBtn =
    document.getElementById(
        "streamPcmbBtn"
    );

const streamPcmBtn =
    document.getElementById(
        "streamPcmBtn"
    );

const previewContainer =
    document.getElementById(
        "previewContainer"
    );

const scanLaserLine =
    document.getElementById(
        "scanLaserLine"
    );

const successState =
    document.getElementById(
        "successState"
    );

const viewResultButton =
    document.getElementById(
        "viewResultButton"
    );

const dashboardSection =
    document.getElementById(
        "dashboardSection"
    );

const classFilter =
    document.getElementById(
        "classFilter"
    );

const sectionFilter =
    document.getElementById(
        "sectionFilter"
    );

const examDashboardFilter =
    document.getElementById(
        "examDashboardFilter"
    );

const dashboardSummary =
    document.getElementById(
        "dashboardSummary"
    );

const dashboardTableBody =
    document.getElementById(
        "dashboardTableBody"
    );

const backToDashboardButton =
    document.getElementById(
        "backToDashboardButton"
    );

const resStudentName =
    document.getElementById(
        "resStudentName"
    );

const resRollNumber =
    document.getElementById(
        "resRollNumber"
    );

const resClassSection =
    document.getElementById(
        "resClassSection"
    );

const resExamDate =
    document.getElementById(
        "resExamDate"
    );

const resSession =
    document.getElementById(
        "resSession"
    );

const questionTableBody =
    document.getElementById(
        "questionTableBody"
    );

const downloadPdfBtn =
    document.getElementById(
        "downloadPdfBtn"
    );

const downloadExcelBtn =
    document.getElementById(
        "downloadExcelBtn"
    );

const downloadWordBtn =
    document.getElementById(
        "downloadWordBtn"
    );

let selectedStream = "pcmb";

let dashboardRows = [];

let latestResultId = null;




/* ==========================================================
   STATE
   ========================================================== */

let cameraStream = null;

let capturedBlob = null;

let previewObjectUrl = null;


/*
    True when current image came from live camera.
    False when image came from file upload.
*/
let capturedFromCamera = false;

let cornerDetectionFrame = null;

let lastCornerCheckAt = 0;

let stableCornerChecks = 0;

let pageCornersDetected = false;

let autoCaptureTriggered = false;

let detectedDocumentBounds = null;

let previousDetection = null;

let consecutiveValidFrames = 0;

const markerAnalysisCanvas = document.createElement(
    "canvas"
);


/* ==========================================================
   CONSTANTS
   ========================================================== */

/*
    A4 portrait:
    210mm x 297mm
*/
const A4_RATIO =
    210 / 297;


/*
    Output image resolution.

    1200px width gives enough detail for OMR
    while keeping upload size manageable.
*/
const CAMERA_OUTPUT_WIDTH =
    1200;


const CAMERA_OUTPUT_HEIGHT =
    Math.round(
        CAMERA_OUTPUT_WIDTH
        / A4_RATIO
    );


const JPEG_QUALITY =
    0.92;


// Require repeated, stable marker recognition before auto-capture.
const AUTO_CAPTURE_STABLE_CHECKS = 4;


/* ==========================================================
   UI HELPERS
   ========================================================== */

function showError(text) {
    if (!errorBox) return;
    errorBox.textContent = text;
    errorBox.hidden = false;
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth" });
}

function clearError() {
    if (!errorBox) return;
    errorBox.textContent = "";
    errorBox.hidden = true;
    errorBox.classList.add("hidden");
}

function showLoading(text = "Processing OMR...") {
    if (!loading) return;
    const txtEl = document.getElementById("loadingText");
    if (txtEl) txtEl.textContent = text;
    else loading.textContent = text;
    loading.hidden = false;
    loading.classList.remove("hidden");
}

function hideLoading() {
    if (!loading) return;
    loading.hidden = true;
    loading.classList.add("hidden");
}

function hideResult() {
    if (!resultSection) return;
    resultSection.hidden = true;
    resultSection.classList.add("hidden");
}

function showSuccessState() {
    if (!successState) return;
    successState.hidden = false;
    successState.classList.remove("hidden");
}

function hideSuccessState() {
    if (!successState) return;
    successState.hidden = true;
    successState.classList.add("hidden");
}

function showDashboard() {
    if (!dashboardSection) return;
    dashboardSection.hidden = false;
    dashboardSection.classList.remove("hidden");
}

function hideDashboard() {
    if (!dashboardSection) return;
    dashboardSection.hidden = true;
    dashboardSection.classList.add("hidden");
}

function renderQuestionTable(questionResults) {
    if (!questionTableBody) return;
    if (!questionResults || Object.keys(questionResults).length === 0) {
        questionTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; padding: 16px; color: var(--muted);">No question details available.</td></tr>`;
        return;
    }

    const qNumbers = Object.keys(questionResults).sort((a, b) => Number(a) - Number(b));
    questionTableBody.innerHTML = qNumbers.map((qNum) => {
        const item = questionResults[qNum] || {};
        const qNo = item.question_number ?? qNum;
        const studentAns = item.detected || item.student_answer || item.answer || "—";
        const correctAns = item.correct_answer || item.correct || "—";
        const rawStatus = (item.status || "Uncertain").toString();
        const statusLower = rawStatus.toLowerCase();

        let badgeClass = "blank";
        if (statusLower.includes("correct") || statusLower.includes("pass")) badgeClass = "correct";
        else if (statusLower.includes("wrong") || statusLower.includes("fail")) badgeClass = "wrong";
        else if (statusLower.includes("multiple")) badgeClass = "multiple";

        return `
            <tr>
                <td><strong>Q${qNo}</strong></td>
                <td>${studentAns}</td>
                <td>${correctAns}</td>
                <td><span class="status-badge ${badgeClass}">${rawStatus}</span></td>
            </tr>
        `;
    }).join("");
}

async function fetchAndRenderDashboard() {
    if (!dashboardTableBody) return;

    dashboardTableBody.innerHTML = `
        <tr>
            <td colspan="12" style="text-align: center; padding: 24px; color: var(--muted);">
                ⚡ Loading evaluated student records from database...
            </td>
        </tr>
    `;

    const classVal = classFilter?.value || "all";
    const sectionVal = sectionFilter?.value || "all";
    const examVal = examDashboardFilter?.value || "all";

    const queryParams = new URLSearchParams();
    if (classVal !== "all") queryParams.append("class", classVal);
    if (sectionVal !== "all") queryParams.append("section", sectionVal);
    if (examVal !== "all") queryParams.append("exam", examVal);

    try {
        const resp = await fetch("/api/omr-results?" + queryParams.toString());
        if (!resp.ok) {
            throw new Error(`Server returned HTTP ${resp.status}`);
        }
        const data = await resp.json();
        dashboardRows = data || [];

        const totalStudents = dashboardRows.length;
        const avgScore = totalStudents
            ? Math.round(dashboardRows.reduce((sum, r) => sum + Number(r.score || 0), 0) / totalStudents)
            : 0;
        const passCount = dashboardRows.filter((r) => Number(r.score || 0) >= 120).length;
        const goodCount = dashboardRows.filter((r) => Number(r.score || 0) >= 90 && Number(r.score || 0) < 120).length;

        if (dashboardSummary) {
            dashboardSummary.innerHTML = `
                <div class="summary-card">
                    <span>Total Students</span>
                    <strong>${totalStudents}</strong>
                </div>
                <div class="summary-card">
                    <span>Avg Score</span>
                    <strong>${avgScore}</strong>
                </div>
                <div class="summary-card">
                    <span>Pass</span>
                    <strong>${passCount}</strong>
                </div>
                <div class="summary-card">
                    <span>Good</span>
                    <strong>${goodCount}</strong>
                </div>
            `;
        }

        if (!dashboardRows.length) {
            dashboardTableBody.innerHTML = `
                <tr>
                    <td colspan="12" style="text-align: center; padding: 24px; color: var(--muted);">
                        No evaluated OMR results found.
                    </td>
                </tr>
            `;
            return;
        }

        dashboardTableBody.innerHTML = dashboardRows.map((r) => {
            const formattedDate = r.date ? new Date(r.date).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) : "-";
            return `
                <tr>
                    <td><strong>${r.student_name || 'Student Candidate'}</strong></td>
                    <td>${r.roll_number || '-'}</td>
                    <td>${r.class || '-'}</td>
                    <td>${r.section || '-'}</td>
                    <td>${r.exam || '-'}</td>
                    <td>${r.paper_code || '-'}</td>
                    <td><strong>${r.score}</strong></td>
                    <td>${r.correct}</td>
                    <td>${r.wrong}</td>
                    <td>${r.blank}</td>
                    <td>${formattedDate}</td>
                    <td>
                        <button type="button" class="action-view-btn" data-id="${r.id}">
                            View
                        </button>
                    </td>
                </tr>
            `;
        }).join("");

        // Attach event listeners to View buttons
        const viewBtns = dashboardTableBody.querySelectorAll(".action-view-btn");
        viewBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                const resId = btn.getAttribute("data-id");
                if (resId) {
                    openIndividualResult(resId);
                }
            });
        });

    } catch (err) {
        console.error("Dashboard fetch error:", err);
        dashboardTableBody.innerHTML = `
            <tr>
                <td colspan="12" style="text-align: center; padding: 24px; color: var(--danger);">
                    Unable to load student results: ${err.message || 'API failure'}
                </td>
            </tr>
        `;
    }
}

function openIndividualResult(resultId) {
    if (!resultId) return;
    window.location.href = `/result.html?id=${resultId}`;
}


function openResultDashboard() {
    clearError();
    hideResult();
    hideSuccessState();
    showDashboard();
    fetchAndRenderDashboard();
    if (dashboardSection) {
        dashboardSection.scrollIntoView({ behavior: "smooth" });
    }
}


/* ==========================================================
   PREVIEW URL CLEANUP
   ========================================================== */

function clearPreviewUrl() {

    if (
        previewObjectUrl
    ) {

        URL.revokeObjectURL(
            previewObjectUrl
        );


        previewObjectUrl =
            null;
    }
}


/* ==========================================================
   CAMERA STOP
   ========================================================== */

function stopCamera() {

    if (cornerDetectionFrame) {

        cancelAnimationFrame(
            cornerDetectionFrame
        );

        cornerDetectionFrame = null;
    }

    stableCornerChecks = 0;

    pageCornersDetected = false;

    autoCaptureTriggered = false;

    detectedDocumentBounds = null;

    previousDetection = null;

    consecutiveValidFrames = 0;

    cameraContainer?.classList.remove(
        "page-corners-detected"
    );

    if (torchEnabled && cameraStream) {
        const track = cameraStream.getVideoTracks()[0];
        if (track) {
            try { track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) { }
        }
    }
    torchEnabled = false;
    updateTorchUI(false, false);

    if (
        cameraStream
    ) {

        const tracks =
            cameraStream.getTracks();


        tracks.forEach(
            function (
                track
            ) {

                track.stop();
            }
        );


        cameraStream =
            null;
    }


    if (
        camera
    ) {

        camera.srcObject =
            null;
    }
}

async function checkTorchSupport(track) {
    if (!track) return false;
    try {
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        const settings = track.getSettings ? track.getSettings() : {};
        return !!capabilities.torch || 'torch' in settings;
    } catch (e) {
        return false;
    }
}

async function toggleTorch() {
    if (!cameraStream) return;
    const track = cameraStream.getVideoTracks()[0];
    if (!track) return;

    try {
        torchEnabled = !torchEnabled;
        await track.applyConstraints({
            advanced: [{ torch: torchEnabled }]
        });
        updateTorchUI(torchEnabled, true);
    } catch (err) {
        console.warn("Torch toggle failed:", err);
        try {
            await track.applyConstraints({ torch: torchEnabled });
            updateTorchUI(torchEnabled, true);
        } catch (e2) {
            showError("Flashlight Error: Not supported on this device/camera.");
            torchEnabled = false;
            updateTorchUI(false, false);
        }
    }
}

function updateTorchUI(active, supported = true) {
    if (!torchButton) return;
    if (!supported) {
        torchButton.classList.add("hidden");
        return;
    }
    torchButton.classList.remove("hidden");
    torchButton.classList.toggle("torch-active", active);
    torchButton.setAttribute("aria-pressed", active ? "true" : "false");
    const icon = torchButton.querySelector(".torch-icon");
    if (icon) {
        icon.textContent = active ? "⚡ Torch ON" : "⚡ Torch OFF";
    }
}


/* ==========================================================
   AUTO-CAPTURE VALIDATION
   ========================================================== */

function isCompleteSheetInFrame(
    sourcePoints,
    videoWidth,
    videoHeight
) {
    /*
        Verify that the entire OMR sheet is visible.

        More lenient than before - only require 2-3% margin
        to match QR scanner speed (doesn't need perfect framing).
    */

    const marginRatio = 0.02; /* Reduced from 0.05 */
    const minX = Math.min(...sourcePoints.map(p => p.x));
    const maxX = Math.max(...sourcePoints.map(p => p.x));
    const minY = Math.min(...sourcePoints.map(p => p.y));
    const maxY = Math.max(...sourcePoints.map(p => p.y));
    const minMargin = Math.min(videoWidth, videoHeight) * marginRatio;

    /* Check that sheet is far enough from edges */
    const leftMargin = minX;
    const rightMargin = videoWidth - maxX;
    const topMargin = minY;
    const bottomMargin = videoHeight - maxY;

    const allMarginsOk = (
        leftMargin >= minMargin &&
        rightMargin >= minMargin &&
        topMargin >= minMargin &&
        bottomMargin >= minMargin
    );

    return allMarginsOk;
}


function isSheetReasonablyAligned(
    sourcePoints
) {
    /*
        Check that the sheet is not severely tilted.
        
        Allow up to 20 degrees tilt (more lenient for QR-scanner speed).
    */

    const topLeft = sourcePoints[0];
    const topRight = sourcePoints[1];

    const dx = topRight.x - topLeft.x;
    const dy = topRight.y - topLeft.y;

    const angleRad = Math.atan2(dy, dx);
    const angleDeg = Math.abs(angleRad * 180 / Math.PI);

    /* Allow up to 20 degrees tilt */
    return angleDeg <= 20;
}


function isSheetLargeEnough(
    sourcePoints,
    videoWidth,
    videoHeight
) {
    /*
        Verify sheet is large enough in frame for reliable recognition.
        
        Reduced requirement for faster capture (15% instead of 25%).
    */

    const minX = Math.min(...sourcePoints.map(p => p.x));
    const maxX = Math.max(...sourcePoints.map(p => p.x));
    const minY = Math.min(...sourcePoints.map(p => p.y));
    const maxY = Math.max(...sourcePoints.map(p => p.y));

    const sheetWidth = maxX - minX;
    const sheetHeight = maxY - minY;
    const sheetArea = sheetWidth * sheetHeight;

    const videoArea = videoWidth * videoHeight;
    const areaRatio = sheetArea / videoArea;

    /* Require at least 15% of frame area */
    return areaRatio >= 0.15;
}


function hasExcessiveMovement(
    currentDetection,
    previousDetection
) {
    /*
        Detect if sheet is moving excessively between frames.
        
        More lenient - allow up to 10% of screen motion (faster capture).
    */

    if (!previousDetection) {
        return false;
    }

    const videoWidth = camera.videoWidth;
    const videoHeight = camera.videoHeight;

    const maxMotion = Math.max(videoWidth, videoHeight) * 0.025;

    const maxDistance = Math.max(
        ...currentDetection.sourcePoints.map((current, idx) => {
            const prev = previousDetection.sourcePoints[idx];
            const dx = current.x - prev.x;
            const dy = current.y - prev.y;
            return Math.sqrt(dx * dx + dy * dy);
        })
    );

    return maxDistance > maxMotion;
}


function isReadyForAutoCapture(
    detection,
    videoWidth,
    videoHeight
) {
    /*
        Camera auto-capture is registration-marker gated.  A white page,
        printed text, bubbles, or a steady camera must never be enough.
    */

    if (!detection || !detection.sourcePoints || detection.markerCount !== 4) {
        return {
            ready: false,
            reason: "Align the OMR sheet — four black corner markers required"
        };
    }

    if (!isSheetLargeEnough(detection.sourcePoints, videoWidth, videoHeight)) {
        return {
            ready: false,
            reason: "Four markers found — move the OMR sheet closer"
        };
    }

    if (!isCompleteSheetInFrame(detection.sourcePoints, videoWidth, videoHeight)) {
        return {
            ready: false,
            reason: "Four markers found — keep the complete sheet inside the frame"
        };
    }

    if (!isSheetReasonablyAligned(detection.sourcePoints)) {
        return {
            ready: false,
            reason: "Four markers found — straighten the OMR sheet"
        };
    }

    if (hasExcessiveMovement(detection, previousDetection)) {
        return {
            ready: false,
            reason: "Four markers found — hold the OMR sheet steady"
        };
    }

    return {
        ready: true,
        reason: "Four black corner markers recognized"
    };
}


/* ==========================================================
   LIVE CORNER-BLOCK DETECTION
   ========================================================== */

function setCornerDetectionState(
    detected,
    statusMessage = null
) {

    pageCornersDetected = detected;

    cameraContainer?.classList.toggle(
        "page-corners-detected",
        detected
    );

    if (cornerDetectionStatus) {

        if (statusMessage) {
            cornerDetectionStatus.textContent = statusMessage;
        } else {
            cornerDetectionStatus.textContent = detected
                ? "OMR detected — hold steady"
                : "Position the complete OMR inside the frame";
        }
    }

    if (captureButton) {

        captureButton.disabled = !detected;
        captureButton.classList.remove("hidden");
    }
}


function findSolidMarkerInZone(
    pixels,
    width,
    height,
    startX,
    startY,
    endX,
    endY
) {
    /*
        Find a compact, filled dark component rather than measuring the total
        amount of dark ink in a large corner region.  This prevents headings,
        table rules, bubbles and other printed content from impersonating a
        registration block.
    */

    const zoneWidth = Math.max(1, endX - startX);
    const zoneHeight = Math.max(1, endY - startY);
    const zoneArea = zoneWidth * zoneHeight;

    let brightnessSum = 0;
    for (let y = startY; y < endY; y += 1) {
        for (let x = startX; x < endX; x += 1) {
            const offset = (y * width + x) * 4;
            brightnessSum +=
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114;
        }
    }

    const averageBrightness = brightnessSum / Math.max(zoneArea, 1);
    const darkThreshold = Math.max(35, Math.min(105, averageBrightness * 0.48));
    const darkMask = new Uint8Array(zoneArea);

    for (let zy = 0; zy < zoneHeight; zy += 1) {
        const y = startY + zy;
        for (let zx = 0; zx < zoneWidth; zx += 1) {
            const x = startX + zx;
            const offset = (y * width + x) * 4;
            const brightness =
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114;
            if (brightness <= darkThreshold) {
                darkMask[zy * zoneWidth + zx] = 1;
            }
        }
    }

    const visited = new Uint8Array(zoneArea);
    const minComponentArea = Math.max(28, Math.round(zoneArea * 0.0015));
    const maxComponentArea = Math.round(zoneArea * 0.30);
    const minSide = Math.max(5, Math.round(Math.min(zoneWidth, zoneHeight) * 0.035));
    const maxSide = Math.round(Math.max(zoneWidth, zoneHeight) * 0.48);

    let best = null;

    for (let index = 0; index < zoneArea; index += 1) {
        if (!darkMask[index] || visited[index]) continue;

        const stack = [index];
        visited[index] = 1;

        let area = 0;
        let minX = zoneWidth;
        let minY = zoneHeight;
        let maxX = -1;
        let maxY = -1;
        let sumX = 0;
        let sumY = 0;

        while (stack.length) {
            const current = stack.pop();
            const cy = Math.floor(current / zoneWidth);
            const cx = current - cy * zoneWidth;

            area += 1;
            sumX += cx;
            sumY += cy;
            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;

            const left = current - 1;
            const right = current + 1;
            const up = current - zoneWidth;
            const down = current + zoneWidth;

            if (cx > 0 && darkMask[left] && !visited[left]) {
                visited[left] = 1;
                stack.push(left);
            }
            if (cx + 1 < zoneWidth && darkMask[right] && !visited[right]) {
                visited[right] = 1;
                stack.push(right);
            }
            if (cy > 0 && darkMask[up] && !visited[up]) {
                visited[up] = 1;
                stack.push(up);
            }
            if (cy + 1 < zoneHeight && darkMask[down] && !visited[down]) {
                visited[down] = 1;
                stack.push(down);
            }
        }

        if (area < minComponentArea || area > maxComponentArea) continue;

        const componentWidth = maxX - minX + 1;
        const componentHeight = maxY - minY + 1;
        if (componentWidth < minSide || componentHeight < minSide) continue;
        if (componentWidth > maxSide || componentHeight > maxSide) continue;

        const aspect = componentWidth / Math.max(componentHeight, 1);
        if (aspect < 0.45 || aspect > 2.2) continue;

        const fill = area / Math.max(componentWidth * componentHeight, 1);
        if (fill < 0.58) continue;

        const squareScore = 1 - Math.min(1, Math.abs(Math.log(aspect)));
        const score = area * fill * (0.65 + 0.35 * squareScore);

        if (!best || score > best.score) {
            best = {
                score,
                area,
                fill,
                aspect,
                x: startX + (sumX / area),
                y: startY + (sumY / area),
            };
        }
    }

    return best;
}


function drawDocumentBoundary(points, detected) {

    if (!documentBoundaryOverlay || !cameraContainer) {

        return;
    }

    const bounds = cameraContainer.getBoundingClientRect();

    const pixelRatio = window.devicePixelRatio || 1;

    const width = Math.max(1, Math.round(bounds.width * pixelRatio));

    const height = Math.max(1, Math.round(bounds.height * pixelRatio));

    if (
        documentBoundaryOverlay.width !== width
        || documentBoundaryOverlay.height !== height
    ) {

        documentBoundaryOverlay.width = width;

        documentBoundaryOverlay.height = height;
    }

    const context = documentBoundaryOverlay.getContext("2d");

    if (!context) {

        return;
    }

    context.setTransform(1, 0, 0, 1, 0, 0);

    context.clearRect(0, 0, width, height);

    if (!points) {

        return;
    }

    context.setTransform(
        pixelRatio,
        0,
        0,
        pixelRatio,
        0,
        0
    );

    context.beginPath();

    points.forEach((point, index) => {
        if (index === 0) {

            context.moveTo(point.x, point.y);
        } else {

            context.lineTo(point.x, point.y);
        }
    });

    context.closePath();

    context.lineWidth = 3;

    // Only actual marker detections are drawn; there is no fixed guide box.
    context.strokeStyle = "#31d57a";

    context.shadowColor = "rgba(0, 0, 0, 0.75)";

    context.shadowBlur = 4;

    context.stroke();

    context.fillStyle = "rgba(49, 213, 122, 0.10)";

    context.fill();

    context.shadowBlur = 0;

    points.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, 5, 0, Math.PI * 2);
        context.fillStyle = "#31d57a";
        context.fill();
    });
}


function detectDocumentCorners() {

    const videoWidth = camera.videoWidth;
    const videoHeight = camera.videoHeight;

    if (!videoWidth || !videoHeight) {
        return null;
    }

    const analysisWidth = 480;
    const analysisHeight = Math.round(
        analysisWidth * (videoHeight / videoWidth)
    );

    markerAnalysisCanvas.width = analysisWidth;
    markerAnalysisCanvas.height = analysisHeight;

    const context = markerAnalysisCanvas.getContext(
        "2d",
        { willReadFrequently: true }
    );

    if (!context) {
        return null;
    }

    context.drawImage(
        camera,
        0,
        0,
        videoWidth,
        videoHeight,
        0,
        0,
        analysisWidth,
        analysisHeight
    );

    const pixels = context.getImageData(
        0,
        0,
        analysisWidth,
        analysisHeight
    ).data;

    /*
        The Manchester templates use four solid registration blocks near the
        sheet corners.  Search only the outer corner regions, then require one
        compact filled component in every region.
    */
    const zoneWidth = Math.round(analysisWidth * 0.32);
    const zoneHeight = Math.round(analysisHeight * 0.27);

    const zones = [
        [0, 0, zoneWidth, zoneHeight],
        [analysisWidth - zoneWidth, 0, analysisWidth, zoneHeight],
        [analysisWidth - zoneWidth, analysisHeight - zoneHeight, analysisWidth, analysisHeight],
        [0, analysisHeight - zoneHeight, zoneWidth, analysisHeight],
    ];

    const markers = zones.map(([sx, sy, ex, ey]) =>
        findSolidMarkerInZone(
            pixels,
            analysisWidth,
            analysisHeight,
            sx,
            sy,
            ex,
            ey
        )
    );

    if (!markers.every(Boolean)) {
        return null;
    }

    const sourcePoints = markers.map(({ x, y }) => ({
        x: (x / analysisWidth) * videoWidth,
        y: (y / analysisHeight) * videoHeight,
    }));

    // Ordered TL, TR, BR, BL.
    const [tl, tr, br, bl] = sourcePoints;

    const top = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const bottom = Math.hypot(br.x - bl.x, br.y - bl.y);
    const left = Math.hypot(bl.x - tl.x, bl.y - tl.y);
    const right = Math.hypot(br.x - tr.x, br.y - tr.y);

    if (Math.min(top, bottom) < videoWidth * 0.42) return null;
    if (Math.min(left, right) < videoHeight * 0.42) return null;

    if (Math.max(top, bottom) / Math.max(Math.min(top, bottom), 1) > 1.75) return null;
    if (Math.max(left, right) / Math.max(Math.min(left, right), 1) > 1.75) return null;

    // Keep the detected quadrilateral large enough to represent a full OMR.
    const polygonArea = Math.abs(
        (tl.x * tr.y - tr.x * tl.y)
        + (tr.x * br.y - br.x * tr.y)
        + (br.x * bl.y - bl.x * br.y)
        + (bl.x * tl.y - tl.x * bl.y)
    ) / 2;

    if (polygonArea < videoWidth * videoHeight * 0.20) {
        return null;
    }

    const displayWidth = cameraContainer.clientWidth;
    const displayHeight = cameraContainer.clientHeight;

    const displayPoints = markers.map(({ x, y }) => ({
        x: (x / analysisWidth) * displayWidth,
        y: (y / analysisHeight) * displayHeight,
    }));

    return {
        displayPoints,
        sourcePoints,
        markerCount: 4,
        markers,
    };
}


function monitorCornerBlocks(timestamp) {

    if (!cameraStream || camera.hidden) {
        cornerDetectionFrame = null;
        return;
    }

    if (timestamp - lastCornerCheckAt >= 250) {
        lastCornerCheckAt = timestamp;

        const detection = detectDocumentCorners();
        const videoWidth = camera.videoWidth;
        const videoHeight = camera.videoHeight;

        const readinessCheck = isReadyForAutoCapture(
            detection,
            videoWidth,
            videoHeight
        );

        if (readinessCheck.ready) {
            stableCornerChecks += 1;
        } else {
            stableCornerChecks = 0;
        }

        const markersStable =
            readinessCheck.ready
            && stableCornerChecks >= AUTO_CAPTURE_STABLE_CHECKS;

        let statusMessage = readinessCheck.reason;

        if (readinessCheck.ready && !markersStable) {
            statusMessage =
                `Four black corner markers recognized — hold steady (${stableCornerChecks}/${AUTO_CAPTURE_STABLE_CHECKS})`;
        }

        setCornerDetectionState(
            Boolean(markersStable),
            markersStable ? "OMR recognized — capturing…" : statusMessage
        );

        drawDocumentBoundary(
            detection?.displayPoints || null,
            Boolean(detection)
        );

        if (markersStable && detection) {
            detectedDocumentBounds = detection.sourcePoints;

            if (!autoCaptureTriggered) {
                autoCaptureTriggered = true;
                setTimeout(() => {
                    captureCameraImage(true);
                }, 120);
            }
        } else {
            detectedDocumentBounds = null;
        }

        previousDetection = detection;
    }

    cornerDetectionFrame = requestAnimationFrame(
        monitorCornerBlocks
    );
}


function cropFromDetectedDocument(
    videoWidth,
    videoHeight
) {
    return {
        x: 0,
        y: 0,
        width: videoWidth,
        height: videoHeight,
    };
}


/* ==========================================================
   CAMERA OPEN
   ========================================================== */

async function openCamera() {

    clearError();

    hideResult();


    if (
        !navigator.mediaDevices
        ||
        !navigator.mediaDevices.getUserMedia
    ) {

        showError(
            "Camera is not available in this browser."
        );

        return;
    }


    stopCamera();

    clearPreviewUrl();


    capturedBlob =
        null;


    capturedFromCamera =
        false;


    try {

        const constraints = {

            audio:
                false,

            video: {

                facingMode: {
                    ideal:
                        "environment"
                },

                width: {
                    ideal:
                        1920
                },

                height: {
                    ideal:
                        1080
                },

                /*
                    Some mobile browsers support
                    continuous autofocus implicitly.
                */
            }
        };


        cameraStream =
            await navigator.mediaDevices
                .getUserMedia(
                    constraints
                );


        camera.setAttribute("playsinline", "true");
        camera.setAttribute("autoplay", "true");
        camera.setAttribute("muted", "true");

        camera.srcObject =
            cameraStream;

        if (documentBoundaryOverlay) {

            documentBoundaryOverlay.hidden = false;
        }


        camera.hidden =
            false;

        camera.classList.remove("hidden");


        capturedPreview.hidden =
            true;


        cameraContainer.hidden =
            false;

        cameraContainer.classList.remove("hidden");


        captureButton.hidden =
            false;

        captureButton.classList.remove("hidden");

        captureButton.disabled =
            true;


        retakeButton.hidden =
            true;

        retakeButton.classList.add("hidden");


        scanButton.disabled =
            true;


        await camera.play();

        const videoTrack = cameraStream.getVideoTracks()[0];
        const torchSupported = await checkTorchSupport(videoTrack);
        updateTorchUI(false, torchSupported);

        lastCornerCheckAt = 0;

        stableCornerChecks = 0;

        setCornerDetectionState(false);

        cornerDetectionFrame = requestAnimationFrame(
            monitorCornerBlocks
        );


    } catch (
    error
    ) {

        console.error(
            error
        );


        showError(
            "Unable to open camera. Allow camera permission and try again."
        );
    }
}


/* ==========================================================
   CALCULATE A4 CROP FROM CAMERA
   ========================================================== */

function calculateA4Crop(
    videoWidth,
    videoHeight
) {
    return {
        x: 0,
        y: 0,
        width: videoWidth,
        height: videoHeight,
    };
}


/* ==========================================================
   CAPTURE CAMERA
   ========================================================== */

function captureCameraImage(
    automatic = false
) {

    if (capturedBlob && !automatic) {
        return;
    }

    clearError();

    hideResult();

    if (
        !cameraStream
    ) {

        showError(
            "Camera is not active."
        );

        return;
    }

    const videoWidth =
        camera.videoWidth;

    const videoHeight =
        camera.videoHeight;

    if (
        !videoWidth
        ||
        !videoHeight
    ) {

        showError(
            "Camera is still starting. Try again."
        );

        return;
    }

    if (!pageCornersDetected) {
        if (cornerDetectionStatus) {
            cornerDetectionStatus.textContent =
                "Align the OMR sheet — four black corner markers must be recognized before capture.";
        }
        showError(
            "Capture blocked: all four black OMR corner markers must be recognized first."
        );
        return;
    }

    if (automatic) {
        autoCaptureTriggered = true;
    }

    captureCanvas.width = videoWidth;
    captureCanvas.height = videoHeight;

    const context =
        captureCanvas.getContext(
            "2d",
            {
                alpha:
                    false
            }
        );

    if (
        !context
    ) {

        showError(
            "Unable to prepare camera image."
        );

        return;
    }

    context.fillStyle =
        "#ffffff";

    context.fillRect(
        0,
        0,
        videoWidth,
        videoHeight
    );

    context.drawImage(

        camera,

        0,
        0,
        videoWidth,
        videoHeight,

        0,
        0,

        videoWidth,
        videoHeight
    );


    captureCanvas.toBlob(

        function (
            blob
        ) {

            if (
                !blob
            ) {

                showError(
                    "Could not capture the camera image."
                );

                return;
            }


            capturedBlob =
                blob;


            capturedFromCamera =
                true;


            clearPreviewUrl();


            previewObjectUrl =
                URL.createObjectURL(
                    blob
                );


            capturedPreview.src =
                previewObjectUrl;

            capturedPreview.hidden =
                false;

            if (previewContainer) {
                previewContainer.classList.remove("hidden");
            }

            if (documentBoundaryOverlay) {

                documentBoundaryOverlay.hidden = true;
            }

            if (cornerDetectionStatus) {

                cornerDetectionStatus.textContent =
                    "Document captured. Select Scan & Evaluate.";
            }

            if (cameraContainer) {
                cameraContainer.classList.add("hidden");
            }

            camera.hidden =
                true;

            captureButton.hidden =
                true;

            retakeButton.hidden =
                false;

            retakeButton.classList.remove("hidden");

            scanButton.hidden =
                false;

            scanButton.classList.remove("hidden");

            scanButton.disabled =
                false;


            /*
                Camera is no longer needed after capture.
            */
            stopCamera();

        },

        "image/jpeg",

        JPEG_QUALITY
    );
}


/* ==========================================================
   PREPARE UPLOADED IMAGE
   ========================================================== */

function prepareUploadedImage(
    file
) {

    clearError();

    hideResult();


    if (
        !file
    ) {

        return;
    }


    const validTypes = [
        "image/jpeg",
        "image/png"
    ];


    if (
        !validTypes.includes(
            file.type
        )
    ) {

        showError(
            "Please upload a JPG or PNG image."
        );

        return;
    }


    capturedBlob = file;
    capturedFromCamera = false;

    clearPreviewUrl();
    previewObjectUrl = URL.createObjectURL(file);

    if (capturedPreview) {
        capturedPreview.src = previewObjectUrl;
        capturedPreview.hidden = false;
    }

    if (previewContainer) {
        previewContainer.classList.remove("hidden");
        previewContainer.hidden = false;
    }

    if (cameraContainer) {
        cameraContainer.classList.add("hidden");
        cameraContainer.hidden = true;
    }

    if (camera) {
        camera.hidden = true;
    }

    if (captureButton) {
        captureButton.hidden = true;
        captureButton.classList.add("hidden");
    }

    if (retakeButton) {
        retakeButton.hidden = false;
        retakeButton.classList.remove("hidden");
    }

    if (scanButton) {
        scanButton.hidden = false;
        scanButton.classList.remove("hidden");
        scanButton.disabled = false;
    }
}


/* ==========================================================
   RETAKE
   ========================================================== */

async function retakeImage() {

    clearError();

    hideResult();


    capturedBlob =
        null;


    capturedFromCamera =
        false;


    clearPreviewUrl();


    capturedPreview.src =
        "";


    capturedPreview.hidden =
        true;


    scanButton.disabled =
        true;


    /*
        If the user originally used camera,
        reopen camera.
 
        Otherwise return to initial state.
    */
    await openCamera();
}


/* ==========================================================
   RESPONSE PARSER
   ========================================================== */

async function readServerResponse(
    response
) {

    const contentType =
        response.headers
            .get(
                "content-type"
            )
        || "";


    const text =
        await response.text();


    let data =
        null;


    if (
        contentType.includes(
            "application/json"
        )
    ) {

        try {

            data =
                JSON.parse(
                    text
                );

        } catch (
        error
        ) {

            console.error(
                "JSON parse error:",
                error
            );
        }
    }


    /*
        Sometimes Vercel / proxy may return JSON
        without correct content-type.
    */
    if (
        !data
        &&
        text
    ) {

        try {

            data =
                JSON.parse(
                    text
                );

        } catch (
        error
        ) {

            /*
                Ignore. We'll produce a readable
                status error below.
            */
        }
    }


    if (
        !response.ok
    ) {

        if (
            data
            &&
            data.detail
        ) {

            throw new Error(
                data.detail
            );
        }


        if (
            data
            &&
            data.error
        ) {

            throw new Error(
                data.error
            );
        }


        if (
            response.status
            === 413
        ) {

            throw new Error(
                "Image is too large for the server."
            );
        }


        if (
            response.status
            === 500
        ) {

            throw new Error(
                "Backend crashed while processing OMR. HTTP 500."
            );
        }


        if (
            response.status
            === 502
        ) {

            throw new Error(
                "Backend service failed. HTTP 502."
            );
        }


        if (
            response.status
            === 504
        ) {

            throw new Error(
                "OMR processing timed out. HTTP 504."
            );
        }


        throw new Error(
            `Server error. HTTP ${response.status}.`
        );
    }


    if (
        !data
    ) {

        throw new Error(
            "Server returned an invalid response."
        );
    }


    return data;
}


/* ==========================================================
   DISPLAY RESULT
   ========================================================== */

function displayResult(
    data,
    showDetailed = true
) {

    const result =
        data.result
        || data;

    // Student & Exam Information Card
    const studentObj = result.student || {};
    const examInfoObj = result.exam_info || {};

    if (resStudentName) {
        resStudentName.textContent = studentObj.name || result.student_name || "Student Candidate";
    }
    if (resRollNumber) {
        resRollNumber.textContent = studentObj.roll_number || result.roll_number || `ROLL-${result.id || result.scan_id || '101'}`;
    }
    if (resClassSection) {
        const cls = studentObj.class || result.class || "12";
        const sec = studentObj.section || result.section || "A";
        resClassSection.textContent = `${cls} - Section ${sec}`;
    }
    if (resExamDate) {
        const rawDt = examInfoObj.exam_date || result.date;
        resExamDate.textContent = rawDt ? new Date(rawDt).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) : "Today";
    }
    if (resSession) {
        resSession.textContent = examInfoObj.session || "Morning";
    }

    // Render Question-wise Analysis
    renderQuestionTable(result.question_results);



    if (
        resultExam
    ) {

        resultExam.textContent =
            (result.exam || result.exam_name || "-").toUpperCase();
    }

    if (
        resultStream
    ) {

        resultStream.textContent =
            (result.stream || selectedStream || "PCMB").toUpperCase();
    }


    if (
        paperCode
    ) {

        paperCode.textContent =
            result.paper_code
            ||
            result.series
            ||
            result.jee_series
            ||
            "-";
    }


    if (
        score
    ) {

        score.textContent =
            result.score
            ?? "-";
    }


    if (
        correct
    ) {

        correct.textContent =
            result.correct
            ?? 0;
    }


    if (
        wrong
    ) {

        wrong.textContent =
            result.wrong
            ?? 0;
    }


    if (
        blank
    ) {

        blank.textContent =
            result.blank
            ?? 0;
    }


    if (
        multiple
    ) {

        multiple.textContent =
            result.multiple
            ?? 0;
    }


    if (
        uncertain
    ) {

        uncertain.textContent =
            result.uncertain
            ?? 0;
    }


    if (
        quality
    ) {

        const qualityData =
            result.quality
            ||
            data.quality;


        if (
            qualityData
        ) {

            quality.textContent =
                `Blur: ${qualityData.blur
                ?? "-"
                } | Brightness: ${qualityData.brightness
                ?? "-"
                } | Contrast: ${qualityData.contrast
                ?? "-"
                }`;

        } else {

            quality.textContent =
                "";
        }
    }


    if (
        message
    ) {

        message.textContent =
            result.message
            ||
            data.message
            ||
            "";
    }


    const correctedUrl =
        result.corrected_image_url
        ||
        data.corrected_image_url;

    if (
        correctedUrl
        &&
        capturedPreview
    ) {
        capturedPreview.src =
            correctedUrl
            + "?t="
            + Date.now();

        capturedPreview.hidden =
            false;
    }


    const bubbleDebugUrl =
        result.bubble_debug_image_url
        ||
        data.bubble_debug_image_url;

    if (
        bubbleDebugUrl
        &&
        bubbleDebugPreview
    ) {
        bubbleDebugPreview.onerror = () => {
            bubbleDebugPreview.removeAttribute("src");
            if (bubbleAnalysisCard) {
                bubbleAnalysisCard.hidden = true;
                bubbleAnalysisCard.classList.add("hidden");
            }
        };

        bubbleDebugPreview.onload = () => {
            if (bubbleAnalysisCard) {
                bubbleAnalysisCard.hidden = false;
                bubbleAnalysisCard.classList.remove("hidden");
            }
        };

        bubbleDebugPreview.src =
            bubbleDebugUrl
            + "?t="
            + Date.now();
    }


    if (
        resultSection
    ) {
        if (showDetailed) {
            resultSection.hidden = false;
            resultSection.classList.remove("hidden");
        } else {
            resultSection.hidden = true;
            resultSection.classList.add("hidden");
        }
    }

    if (!showDetailed) {
        showSuccessState();
        hideDashboard();
    }
}



/* ==========================================================
   SCAN
   ========================================================== */

async function scanOMR() {

    clearError();

    hideResult();
    hideSuccessState();
    hideDashboard();

    const exam =
        examSelect
            ?.value
            ?.trim();


    if (
        !exam
    ) {

        showError(
            "Select an exam first."
        );

        return;
    }


    if (
        !capturedBlob
    ) {

        showError(
            "Capture or upload an OMR image first."
        );

        return;
    }


    scanButton.disabled =
        true;

    if (scanLaserLine) {
        scanLaserLine.classList.remove("hidden");
    }

    showLoading(
        "⚡ Scanning & Evaluating OMR Sheet... Analyzing Bubbles... Please Wait..."
    );

    if (loading) {
        loading.scrollIntoView({ behavior: "smooth" });
    }


    try {

        const formData =
            new FormData();


        formData.append(
            "exam",
            exam
        );

        formData.append(
            "stream",
            exam === "kcet" ? selectedStream : "pcmb"
        );

        formData.append(
            "image",
            capturedBlob,
            capturedFromCamera
                ? "camera_omr.jpg"
                // Keep the original file bytes and name.  The server owns
                // the one deterministic decode/EXIF normalization pass;
                // re-encoding here would introduce device-dependent JPEG
                // artifacts before recognition.
                : (capturedBlob.name || "uploaded_omr.jpg")
        );


        const response =
            await fetch(
                "/scan",
                {
                    method:
                        "POST",

                    body:
                        formData
                }
            );


        const data =
            await readServerResponse(
                response
            );

        latestResultId = data?.id || data?.scan_id || null;

        displayResult(
            data,
            false
        );

        showSuccessState();
        hideResult();
        hideDashboard();
        if (successState) {
            successState.scrollIntoView({ behavior: "smooth" });
        }



    } catch (
    error
    ) {

        console.error(
            error
        );


        showError(
            error.message
            ||
            "OMR processing failed."
        );


    } finally {

        if (scanLaserLine) {
            scanLaserLine.classList.add("hidden");
        }

        hideLoading();


        scanButton.disabled =
            false;
    }
}


/* ==========================================================
   EVENTS
   ========================================================== */

if (
    openCameraButton
) {

    openCameraButton.addEventListener(
        "click",
        openCamera
    );
}


if (
    captureButton
) {

    captureButton.addEventListener(
        "click",
        captureCameraImage
    );
}


if (
    retakeButton
) {

    retakeButton.addEventListener(
        "click",
        retakeImage
    );
}


if (
    scanButton
) {

    scanButton.addEventListener(
        "click",
        scanOMR
    );
}

if (
    torchButton
) {

    torchButton.addEventListener(
        "click",
        toggleTorch
    );
}


if (
    imageUpload
) {

    imageUpload.addEventListener(
        "change",
        function (
            event
        ) {

            const file =
                event.target
                    .files?.[0];


            if (
                file
            ) {

                stopCamera();

                prepareUploadedImage(
                    file
                );
            }

            hideSuccessState();
            hideDashboard();

            /*
                Allows selecting the same image again.
            */
            event.target.value =
                "";
        }
    );
}

if (viewResultButton) {
    viewResultButton.addEventListener("click", () => {
        if (!latestResultId) {
            showError("Result ID is not available. Please scan the OMR sheet again.");
            return;
        }
        openIndividualResult(latestResultId);
    });
}

if (backToDashboardButton) {
    backToDashboardButton.addEventListener("click", () => {
        window.location.href = "/dashboard";
    });
}

if (classFilter) {
    classFilter.addEventListener("change", fetchAndRenderDashboard);
}

if (sectionFilter) {
    sectionFilter.addEventListener("change", fetchAndRenderDashboard);
}

if (examDashboardFilter) {
    examDashboardFilter.addEventListener("change", fetchAndRenderDashboard);
}

/* DOWNLOAD HANDLERS */
function downloadPDF() {
    window.print();
}

function downloadExcel() {
    const name = (resStudentName?.textContent || "Student").replace(/[^a-zA-Z0-9]/g, "_");
    const roll = (resRollNumber?.textContent || "101").replace(/[^a-zA-Z0-9]/g, "_");

    let csv = "STUDENT EVALUATION REPORT\n";
    csv += `Student Name,${resStudentName?.textContent || '-'}\n`;
    csv += `Roll Number,${resRollNumber?.textContent || '-'}\n`;
    csv += `Class & Section,${resClassSection?.textContent || '-'}\n`;
    csv += `Exam,${resultExam?.textContent || '-'}\n`;
    csv += `Stream,${resultStream?.textContent || '-'}\n`;
    csv += `Paper / Series,${paperCode?.textContent || '-'}\n`;
    csv += `Total Score,${score?.textContent || '-'}\n`;
    csv += `Correct,${correct?.textContent || '0'}\n`;
    csv += `Wrong,${wrong?.textContent || '0'}\n`;
    csv += `Blank,${blank?.textContent || '0'}\n`;
    csv += `Multiple,${multiple?.textContent || '0'}\n`;
    csv += `Uncertain,${uncertain?.textContent || '0'}\n\n`;

    csv += "QUESTION-WISE ANALYSIS\n";
    csv += "Question #,Student Answer,Correct Answer,Status\n";

    if (questionTableBody) {
        const trs = questionTableBody.querySelectorAll("tr");
        trs.forEach(tr => {
            const tds = tr.querySelectorAll("td");
            if (tds.length === 4) {
                const q = tds[0].textContent.trim();
                const s = tds[1].textContent.trim();
                const c = tds[2].textContent.trim();
                const st = tds[3].textContent.trim();
                csv += `"${q}","${s}","${c}","${st}"\n`;
            }
        });
    }

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `OMR_Report_${name}_${roll}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function downloadWord() {
    const name = (resStudentName?.textContent || "Student").replace(/[^a-zA-Z0-9]/g, "_");
    const roll = (resRollNumber?.textContent || "101").replace(/[^a-zA-Z0-9]/g, "_");

    let html = `<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
    <head><title>OMR Evaluation Report</title><style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    h2 { color: #1a365d; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    th, td { border: 1px solid #cbd5e1; padding: 8px; text-align: left; }
    th { background-color: #f1f5f9; }
    </style></head><body>
    <h2>OMR Evaluation Report — Manchester Technologies</h2>
    <hr/>
    <h3>Student Information</h3>
    <p><b>Student Name:</b> ${resStudentName?.textContent || '-'}</p>
    <p><b>Roll Number:</b> ${resRollNumber?.textContent || '-'}</p>
    <p><b>Class & Section:</b> ${resClassSection?.textContent || '-'}</p>
    <p><b>Exam:</b> ${resultExam?.textContent || '-'}</p>
    <p><b>Stream:</b> ${resultStream?.textContent || '-'}</p>
    <p><b>Paper / Series:</b> ${paperCode?.textContent || '-'}</p>
    
    <h3>Score Summary</h3>
    <table>
        <tr><th>Total Score</th><th>Correct</th><th>Wrong</th><th>Blank</th><th>Multiple</th><th>Uncertain</th></tr>
        <tr>
            <td><b>${score?.textContent || '-'}</b></td>
            <td>${correct?.textContent || '0'}</td>
            <td>${wrong?.textContent || '0'}</td>
            <td>${blank?.textContent || '0'}</td>
            <td>${multiple?.textContent || '0'}</td>
            <td>${uncertain?.textContent || '0'}</td>
        </tr>
    </table>

    <h3>Detailed Evaluation Breakdown Per Question</h3>
    <table>
        <thead>
            <tr><th>Question #</th><th>Student Answer</th><th>Correct Answer</th><th>Status</th></tr>
        </thead>
        <tbody>`;

    if (questionTableBody) {
        const trs = questionTableBody.querySelectorAll("tr");
        trs.forEach(tr => {
            const tds = tr.querySelectorAll("td");
            if (tds.length === 4) {
                html += `<tr><td>${tds[0].innerHTML}</td><td>${tds[1].innerHTML}</td><td>${tds[2].innerHTML}</td><td>${tds[3].innerHTML}</td></tr>`;
            }
        });
    }

    html += `</tbody></table></body></html>`;

    const blob = new Blob([html], { type: "application/msword" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `OMR_Report_${name}_${roll}.doc`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

if (downloadPdfBtn) {
    downloadPdfBtn.addEventListener("click", downloadPDF);
}

if (downloadExcelBtn) {
    downloadExcelBtn.addEventListener("click", downloadExcel);
}

if (downloadWordBtn) {
    downloadWordBtn.addEventListener("click", downloadWord);
}



function updateStreamUI(stream) {
    selectedStream = stream.toLowerCase();
    if (streamPcmbBtn && streamPcmBtn) {
        if (selectedStream === "pcm") {
            streamPcmbBtn.classList.remove("active");
            streamPcmBtn.classList.add("active");
        } else {
            streamPcmBtn.classList.remove("active");
            streamPcmbBtn.classList.add("active");
        }
    }
}

if (streamPcmbBtn) {
    streamPcmbBtn.addEventListener("click", function () {
        updateStreamUI("pcmb");
    });
}

if (streamPcmBtn) {
    streamPcmBtn.addEventListener("click", function () {
        updateStreamUI("pcm");
    });
}

function updateExamStreamVisibility() {
    const selected = examSelect?.value?.toLowerCase()?.trim();
    if (kcetStreamSection) {
        if (selected === "kcet") {
            kcetStreamSection.classList.remove("hidden");
        } else {
            kcetStreamSection.classList.add("hidden");
        }
    }
}

if (examSelect) {
    examSelect.addEventListener("change", updateExamStreamVisibility);
}

/* LIGHTBOX FULLSCREEN ZOOM */
const imageLightbox = document.getElementById("imageLightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxClose = document.getElementById("lightboxClose");

function openLightbox(src) {
    if (imageLightbox && lightboxImg && src) {
        lightboxImg.src = src;
        imageLightbox.classList.remove("hidden");
    }
}

function closeLightbox() {
    if (imageLightbox) {
        imageLightbox.classList.add("hidden");
    }
}

if (bubbleDebugPreview) {
    bubbleDebugPreview.addEventListener("click", () => {
        openLightbox(bubbleDebugPreview.src);
    });
}

if (capturedPreview) {
    capturedPreview.style.cursor = "zoom-in";
    capturedPreview.addEventListener("click", () => {
        openLightbox(capturedPreview.src);
    });
}

if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
}

if (imageLightbox) {
    imageLightbox.addEventListener("click", (e) => {
        if (e.target === imageLightbox) {
            closeLightbox();
        }
    });
}


/* ==========================================================
   PAGE CLEANUP
   ========================================================== */

window.addEventListener(
    "beforeunload",
    function () {

        stopCamera();

        clearPreviewUrl();
    }
);


/* ==========================================================
   INITIAL UI
   ========================================================== */

(function initializeUI() {

    hideLoading();

    clearError();

    hideResult();
    hideSuccessState();
    hideDashboard();

    if (
        cameraContainer
    ) {

        cameraContainer.hidden =
            true;
    }


    if (
        camera
    ) {

        camera.hidden =
            false;
    }


    if (
        capturedPreview
    ) {

        capturedPreview.hidden =
            true;
    }


    if (
        captureButton
    ) {

        captureButton.hidden =
            true;
    }


    if (
        retakeButton
    ) {

        retakeButton.hidden =
            true;
    }


    if (
        scanButton
    ) {

        scanButton.disabled =
            true;
    }
})();
