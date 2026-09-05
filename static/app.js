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

const jeeMetadataSection =
    document.getElementById(
        "jeeMetadataSection"
    );

const jeeClass =
    document.getElementById(
        "jeeClass"
    );

const jeeSection =
    document.getElementById(
        "jeeSection"
    );

const jeeExamDate =
    document.getElementById(
        "jeeExamDate"
    );

const jeeSession =
    document.getElementById(
        "jeeSession"
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

let batchUploadFiles = [];
const MAX_BATCH_OMR_FILES = 500;

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

let cameraFocusWarmupUntil = 0;

if (imageUpload) {
    imageUpload.addEventListener(
        "change",
        (event) => {
            const files = Array.from(
                event.target.files || []
            );

            if (files.length > MAX_BATCH_OMR_FILES) {
                batchUploadFiles = [];
                event.target.value = "";

                showError(
                    "You can upload a maximum of "
                    + MAX_BATCH_OMR_FILES
                    + " OMR images at one time."
                );

                return;
            }

            batchUploadFiles = files;
        },
        true
    );
}

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
    0.97;


// Two consecutive checks are still effectively instant for the user, while
// preventing a single blurred frame from triggering capture.
const AUTO_CAPTURE_STABLE_CHECKS = 1;
const AUTO_CAPTURE_CHECK_INTERVAL_MS = 45;

// Downscaled live-frame Laplacian variance.
// A blurred frame must never trigger automatic capture.
const AUTO_CAPTURE_MIN_SHARPNESS = 1400;



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
        const studentAns = item.detected || item.student_answer || item.answer || "Ã¢â‚¬â€";
        const correctAns = item.correct_answer || item.correct || "Ã¢â‚¬â€";
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
                Ã¢Å¡Â¡ Loading evaluated student records from database...
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
                        <button type="button" class="action-view-btn" data-id="${r.scan_id || r.id}">
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

function openIndividualResult(resultKey) {
    if (!resultKey) return;
    window.location.href = `/result.html?id=${encodeURIComponent(resultKey)}`;
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

    cameraFocusWarmupUntil = 0;

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
        icon.textContent = active ? "Ã¢Å¡Â¡ Torch ON" : "Ã¢Å¡Â¡ Torch OFF";
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


function estimateFrameSharpness(
    pixels,
    width,
    height
) {
    if (
        !pixels
        || width < 40
        || height < 40
    ) {
        return 0;
    }

    const left = Math.max(
        2,
        Math.floor(width * 0.08)
    );
    const right = Math.min(
        width - 3,
        Math.ceil(width * 0.92)
    );
    const top = Math.max(
        2,
        Math.floor(height * 0.08)
    );
    const bottom = Math.min(
        height - 3,
        Math.ceil(height * 0.92)
    );

    function grayAt(x, y) {
        const index = (
            (y * width + x)
            * 4
        );

        return (
            pixels[index] * 0.299
            + pixels[index + 1] * 0.587
            + pixels[index + 2] * 0.114
        );
    }

    let sum = 0;
    let sumSquares = 0;
    let count = 0;

    for (
        let y = top;
        y <= bottom;
        y += 2
    ) {
        for (
            let x = left;
            x <= right;
            x += 2
        ) {
            const center = grayAt(x, y);

            const laplacian = (
                4 * center
                - grayAt(x - 1, y)
                - grayAt(x + 1, y)
                - grayAt(x, y - 1)
                - grayAt(x, y + 1)
            );

            sum += laplacian;
            sumSquares += (
                laplacian
                * laplacian
            );
            count += 1;
        }
    }

    if (!count) {
        return 0;
    }

    const mean = (
        sum
        / count
    );

    return Math.max(
        0,
        (
            sumSquares
            / count
        )
        - mean * mean
    );
}


function isReadyForAutoCapture(
    detection,
    videoWidth,
    videoHeight,
    priorDetection
) {
    /*
        CORNER-ONLY AUTOCAPTURE CONTRACT

        Automatic capture depends ONLY on detection of all four Manchester
        black registration blocks.

        No bubble state, focus score, page margin, page size, movement,
        tilt, or alignment check is allowed to delay automatic capture here.

        detectDocumentCorners() itself still verifies that the candidates are
        compact square markers in TL/TR/BR/BL and form a plausible outer-page
        quadrilateral.  That is the only automatic-capture gate.
    */
    if (
        detection
        && detection.sourcePoints
        && detection.markerCount === 4
    ) {
        return {
            ready: true,
            reason: "Four black corner blocks detected — capturing…"
        };
    }

    return {
        ready: false,
        reason: "Show all four black corner blocks"
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
                ? "OMR detected Ã¢â‚¬â€ hold steady"
                : "Position the complete OMR inside the frame";
        }
    }

    if (captureButton) {
        // Manual capture must always remain available while the camera is
        // active. Only automatic capture is registration-box gated.
        captureButton.disabled = !cameraStream;
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
    endY,
    corner
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

    // Keep only pixels with a genuinely solid 3x3 neighbourhood. This erodes
    // thin printed borders/letters and separates a corner square from the page
    // border it touches, while preserving the square's filled interior.
    const solidMask = new Uint8Array(zoneArea);
    for (let zy = 1; zy + 1 < zoneHeight; zy += 1) {
        for (let zx = 1; zx + 1 < zoneWidth; zx += 1) {
            let darkNeighbours = 0;
            for (let oy = -1; oy <= 1; oy += 1) {
                for (let ox = -1; ox <= 1; ox += 1) {
                    darkNeighbours += darkMask[(zy + oy) * zoneWidth + zx + ox];
                }
            }
            if (darkNeighbours === 9) {
                solidMask[zy * zoneWidth + zx] = 1;
            }
        }
    }

    const visited = new Uint8Array(zoneArea);
    // Corner blocks become only 3-8 analysis pixels wide when an A4 portrait
    // page is held inside a landscape phone-camera frame. Keep the lower
    // limits scale-safe while rejecting large logos and headings.
    const minComponentArea = Math.max(4, Math.round(zoneArea * 0.00003));
    const maxComponentArea = Math.round(zoneArea * 0.08);
    const minSide = 3;
    const maxSide = Math.max(12, Math.round(Math.min(zoneWidth, zoneHeight) * 0.28));

    let best = null;

    for (let index = 0; index < zoneArea; index += 1) {
        if (!solidMask[index] || visited[index]) continue;

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

            if (cx > 0 && solidMask[left] && !visited[left]) {
                visited[left] = 1;
                stack.push(left);
            }
            if (cx + 1 < zoneWidth && solidMask[right] && !visited[right]) {
                visited[right] = 1;
                stack.push(right);
            }
            if (cy > 0 && solidMask[up] && !visited[up]) {
                visited[up] = 1;
                stack.push(up);
            }
            if (cy + 1 < zoneHeight && solidMask[down] && !visited[down]) {
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
        if (aspect < 0.50 || aspect > 1.90) continue;

        const fill = area / Math.max(componentWidth * componentHeight, 1);
        if (fill < 0.65) continue;

        // A filled response bubble is circular: the four corners of its
        // bounding box remain mostly white. A registration mark is a solid
        // square and keeps dark ink in every bounding-box corner. This is the
        // primary guard against bubbles or printed letters triggering capture.
        const cornerSizeX = Math.max(1, Math.floor(componentWidth * 0.28));
        const cornerSizeY = Math.max(1, Math.floor(componentHeight * 0.28));
        const componentCorners = [
            [minX, minY],
            [maxX - cornerSizeX + 1, minY],
            [maxX - cornerSizeX + 1, maxY - cornerSizeY + 1],
            [minX, maxY - cornerSizeY + 1],
        ];
        const cornerOccupancies = componentCorners.map(([cornerX, cornerY]) => {
            let dark = 0;
            let total = 0;
            for (let yy = cornerY; yy < cornerY + cornerSizeY; yy += 1) {
                for (let xx = cornerX; xx < cornerX + cornerSizeX; xx += 1) {
                    if (xx < 0 || xx >= zoneWidth || yy < 0 || yy >= zoneHeight) continue;
                    dark += solidMask[yy * zoneWidth + xx] ? 1 : 0;
                    total += 1;
                }
            }
            return dark / Math.max(total, 1);
        });
        const minimumCornerOccupancy = Math.min(...cornerOccupancies);
        const averageCornerOccupancy = (
            cornerOccupancies.reduce((sum, value) => sum + value, 0)
            / cornerOccupancies.length
        );
        // Perspective and blur can clip one physical square corner, especially
        // lower registration boxes touching the printed border. A round
        // response bubble still has low occupancy across MOST bounding-box
        // corners, so use the four-corner average.

        const squareScore = 1 - Math.min(1, Math.abs(Math.log(aspect)));
        const centerX = startX + (sumX / area);
        const centerY = startY + (sumY / area);
        const target = {
            TL: { x: 0, y: 0 },
            TR: { x: width, y: 0 },
            BR: { x: width, y: height },
            BL: { x: 0, y: height },
        }[corner];
        const cornerDistance = Math.hypot(
            (centerX - target.x) / Math.max(width, 1),
            (centerY - target.y) / Math.max(height, 1)
        );
        const sizeQuality = Math.min(area, 225);
        const score = (
            sizeQuality
            * fill
            * (0.55 + 0.45 * squareScore)
            * (0.60 + 0.40 * averageCornerOccupancy)
            / (0.08 + cornerDistance)
        );

        if (!best || score > best.score) {
            best = {
                score,
                area,
                fill,
                aspect,
                componentWidth,
                componentHeight,
                cornerOccupancy: averageCornerOccupancy,
                x: centerX,
                y: centerY,
            };
        }
    }

    return best;
}


function buildBrightnessIntegral(pixels, width, height) {
    const stride = width + 1;
    const integral = new Float64Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y += 1) {
        let rowSum = 0;
        for (let x = 0; x < width; x += 1) {
            const offset = (y * width + x) * 4;
            rowSum += (
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114
            );
            integral[(y + 1) * stride + x + 1] =
                integral[y * stride + x + 1] + rowSum;
        }
    }
    return { integral, stride, width, height };
}


function integralBoxMean(integralData, centerX, centerY, side) {
    const { integral, stride, width, height } = integralData;
    const half = Math.floor(side / 2);
    const x0 = Math.max(0, centerX - half);
    const y0 = Math.max(0, centerY - half);
    const x1 = Math.min(width, centerX + side - half);
    const y1 = Math.min(height, centerY + side - half);
    const sum = (
        integral[y1 * stride + x1]
        - integral[y0 * stride + x1]
        - integral[y1 * stride + x0]
        + integral[y0 * stride + x0]
    );
    return sum / Math.max((x1 - x0) * (y1 - y0), 1);
}


function findSolidSquareByContrast(
    integralData,
    startX,
    startY,
    endX,
    endY,
    corner
) {
    const { width, height } = integralData;
    const target = {
        TL: { x: 0, y: 0 },
        TR: { x: width, y: 0 },
        BR: { x: width, y: height },
        BL: { x: 0, y: height },
    }[corner];
    const minSide = Math.max(4, Math.round(width * 0.006));
    const maxSide = Math.max(12, Math.round(width * 0.030));
    const sideStep = Math.max(2, Math.round(width * 0.003));
    const positionStep = 3;
    let best = null;

    for (let side = minSide; side <= maxSide; side += sideStep) {
        const outerSide = side * 2 + 1;
        const edgeMargin = Math.ceil(outerSide / 2);
        for (
            let y = Math.max(startY, edgeMargin);
            y < Math.min(endY, height - edgeMargin);
            y += positionStep
        ) {
            for (
                let x = Math.max(startX, edgeMargin);
                x < Math.min(endX, width - edgeMargin);
                x += positionStep
            ) {
                const innerMean = integralBoxMean(integralData, x, y, side);
                const outerMean = integralBoxMean(integralData, x, y, outerSide);
                const contrast = outerMean - innerMean;
                if (contrast < 8 || innerMean > 185) continue;

                // Solid squares remain dark in all four inner corners. A
                // circular filled bubble becomes bright in at least one.
                const cornerOffset = Math.max(1, Math.round(side * 0.30));
                const cornerSide = Math.max(2, Math.round(side * 0.28));
                const cornerMeans = [
                    integralBoxMean(integralData, x - cornerOffset, y - cornerOffset, cornerSide),
                    integralBoxMean(integralData, x + cornerOffset, y - cornerOffset, cornerSide),
                    integralBoxMean(integralData, x + cornerOffset, y + cornerOffset, cornerSide),
                    integralBoxMean(integralData, x - cornerOffset, y + cornerOffset, cornerSide),
                ];
                if (Math.max(...cornerMeans) > innerMean + 52) continue;

                const distance = Math.hypot(
                    (x - target.x) / Math.max(width, 1),
                    (y - target.y) / Math.max(height, 1)
                );
                const darkness = (180 - Math.min(innerMean, 180)) / 180;
                const score = contrast / 80 + darkness * 0.25 - distance * 0.80;
                if (!best || score > best.score) {
                    best = {
                        score,
                        x,
                        y,
                        componentWidth: side,
                        componentHeight: side,
                        fill: 1,
                        aspect: 1,
                        cornerOccupancy: 1,
                        contrast,
                        meanBrightness: innerMean,
                    };
                }
            }
        }
    }

    return best && best.score > 0.20 ? best : null;
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

    // At 480px a registration block can shrink below three pixels when a
    // portrait sheet is viewed by a landscape phone sensor. 640px preserves
    // enough of the square for reliable component/contrast detection while
    // remaining small enough for live analysis on mobile devices.
    const analysisWidth = Math.min(640, videoWidth);
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
    // A portrait A4 page inside a landscape camera can start around 30% of
    // the frame width. Search nearly the whole quadrant, then rank compact
    // candidates toward its true frame corner.
    const zoneWidth = Math.round(analysisWidth * 0.46);
    const zoneHeight = Math.round(analysisHeight * 0.46);

    const zones = [
        [0, 0, zoneWidth, zoneHeight],
        [analysisWidth - zoneWidth, 0, analysisWidth, zoneHeight],
        [analysisWidth - zoneWidth, analysisHeight - zoneHeight, analysisWidth, analysisHeight],
        [0, analysisHeight - zoneHeight, zoneWidth, analysisHeight],
    ];

    const cornerNames = ["TL", "TR", "BR", "BL"];
    let markers = zones.map(([sx, sy, ex, ey], index) =>
        findSolidMarkerInZone(
            pixels,
            analysisWidth,
            analysisHeight,
            sx,
            sy,
            ex,
            ey,
            cornerNames[index]
        )
    );

    // STRICT REGISTRATION-BLOCK CONTRACT:
    // All four markers must be found by the solid-component square detector.
    // Do not synthesize a missing marker from generic dark contrast patches.
    // This prevents OMR bubbles, text, logos, or table cells from triggering
    // automatic capture.
    if (!markers.every(Boolean)) {
        return null;
    }



    const sourcePoints = markers.map(({ x, y }) => ({
        x: (x / analysisWidth) * videoWidth,
        y: (y / analysisHeight) * videoHeight,
    }));

    // Ordered TL, TR, BR, BL.
    const [tl, tr, br, bl] = sourcePoints;

    // Registration blocks must occupy the four *outer* page corners.  This
    // geometry guard prevents four dark response bubbles, table cells, or
    // logo fragments from forming a plausible but incorrect quadrilateral.
    const outerCornerGeometry = (
        tl.x <= videoWidth * 0.43
        && bl.x <= videoWidth * 0.43
        && tr.x >= videoWidth * 0.57
        && br.x >= videoWidth * 0.57
        && tl.y <= videoHeight * 0.38
        && tr.y <= videoHeight * 0.38
        && bl.y >= videoHeight * 0.62
        && br.y >= videoHeight * 0.62
    );
    if (!outerCornerGeometry) return null;

    // The two left markers and the two right markers must describe the same
    // page edges.  Bubbles selected from different answer columns fail this
    // test even when each individual component happens to look dark/square.
    if (Math.abs(tl.x - bl.x) > videoWidth * 0.22) return null;
    if (Math.abs(tr.x - br.x) > videoWidth * 0.22) return null;
    if (Math.abs(tl.y - tr.y) > videoHeight * 0.18) return null;
    if (Math.abs(bl.y - br.y) > videoHeight * 0.18) return null;

    // All four printed registration boxes use the same physical dimensions.
    // Perspective can change their apparent size, but a bubble/text candidate
    // mixed with real boxes produces a much larger inconsistency.
    const markerSides = markers.map(marker =>
        Math.sqrt(marker.componentWidth * marker.componentHeight)
    );
    const smallestMarkerSide = Math.min(...markerSides);
    const largestMarkerSide = Math.max(...markerSides);
    if (largestMarkerSide / Math.max(smallestMarkerSide, 1) > 1.95) {
        return null;
    }

    const top = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const bottom = Math.hypot(br.x - bl.x, br.y - bl.y);
    const left = Math.hypot(bl.x - tl.x, bl.y - tl.y);
    const right = Math.hypot(br.x - tr.x, br.y - tr.y);

    if (Math.min(top, bottom) < videoWidth * 0.34) return null;
    if (Math.min(left, right) < videoHeight * 0.34) return null;

    if (Math.max(top, bottom) / Math.max(Math.min(top, bottom), 1) > 1.75) return null;
    if (Math.max(left, right) / Math.max(Math.min(left, right), 1) > 1.75) return null;

    const observedSheetRatio = (
        (top + bottom) / Math.max(left + right, 1)
    );
    if (observedSheetRatio < 0.42 || observedSheetRatio > 1.05) {
        return null;
    }

    // Keep the detected quadrilateral large enough to represent a full OMR.
    const polygonArea = Math.abs(
        (tl.x * tr.y - tr.x * tl.y)
        + (tr.x * br.y - br.x * tr.y)
        + (br.x * bl.y - bl.x * br.y)
        + (bl.x * tl.y - tl.x * bl.y)
    ) / 2;

    if (polygonArea < videoWidth * videoHeight * 0.12) {
        return null;
    }

    const averageSheetWidth = (top + bottom) / 2;
    const averageMarkerSide = (
        markerSides.reduce((sum, value) => sum + value, 0)
        / markerSides.length
    ) * (videoWidth / analysisWidth);
    const markerToSheetRatio = averageMarkerSide / Math.max(averageSheetWidth, 1);
    if (markerToSheetRatio < 0.009 || markerToSheetRatio > 0.038) {
        return null;
    }

    const displayWidth = cameraContainer.clientWidth;
    const displayHeight = cameraContainer.clientHeight;

    const displayPoints = markers.map(({ x, y }) => ({
        x: (x / analysisWidth) * displayWidth,
        y: (y / analysisHeight) * displayHeight,
    }));

    const sharpness = estimateFrameSharpness(
        pixels,
        analysisWidth,
        analysisHeight
    );

    return {
        displayPoints,
        sourcePoints,
        markerCount: 4,
        markers,
        sharpness,
    };
}


function monitorCornerBlocks(timestamp) {

    if (!cameraStream || camera.hidden) {
        cornerDetectionFrame = null;
        return;
    }

    if (timestamp - lastCornerCheckAt >= AUTO_CAPTURE_CHECK_INTERVAL_MS) {
        lastCornerCheckAt = timestamp;

        const detection = detectDocumentCorners();
        const videoWidth = camera.videoWidth;
        const videoHeight = camera.videoHeight;

        const readinessCheck = isReadyForAutoCapture(
            detection,
            videoWidth,
            videoHeight,
            previousDetection
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
                `Four black corner markers recognized Ã¢â‚¬â€ hold steady (${stableCornerChecks}/${AUTO_CAPTURE_STABLE_CHECKS})`;
        }

        setCornerDetectionState(
            Boolean(markersStable),
            markersStable ? "OMR recognized Ã¢â‚¬â€ capturingÃ¢â‚¬Â¦" : statusMessage
        );

        drawDocumentBoundary(
            detection?.displayPoints || null,
            Boolean(detection)
        );

        if (markersStable && detection) {
            detectedDocumentBounds = detection.sourcePoints;

            if (!autoCaptureTriggered) {
                autoCaptureTriggered = true;
                captureCameraImage(true);
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
                        2560
                },

                height: {
                    ideal:
                        1440
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

        if (captureButton) {
            captureButton.disabled = false;
            captureButton.classList.remove("hidden");
        }

        // Match the preview/overlay box to the actual sensor frame. This
        // prevents CSS from showing a cropped image while detection and the
        // saved JPEG operate on the complete frame.
        if (camera.videoWidth && camera.videoHeight) {
            cameraContainer.style.aspectRatio =
                `${camera.videoWidth} / ${camera.videoHeight}`;
        }

        const videoTrack = cameraStream.getVideoTracks()[0];

        if (videoTrack) {
            try {
                const capabilities =
                    typeof videoTrack.getCapabilities === "function"
                        ? videoTrack.getCapabilities()
                        : {};

                const advanced = {};

                if (
                    Array.isArray(capabilities.focusMode)
                    && capabilities.focusMode.includes("continuous")
                ) {
                    advanced.focusMode = "continuous";
                }

                if (
                    Array.isArray(capabilities.exposureMode)
                    && capabilities.exposureMode.includes("continuous")
                ) {
                    advanced.exposureMode = "continuous";
                }

                if (
                    Array.isArray(capabilities.whiteBalanceMode)
                    && capabilities.whiteBalanceMode.includes("continuous")
                ) {
                    advanced.whiteBalanceMode = "continuous";
                }

                if (Object.keys(advanced).length) {
                    await videoTrack.applyConstraints({
                        advanced: [advanced]
                    });
                }
            } catch (focusError) {
                console.debug(
                    "Continuous camera focus/exposure unavailable:",
                    focusError
                );
            }
        }

        cameraFocusWarmupUntil =
            performance.now() + 350;

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

    if (!automatic) {
        autoCaptureTriggered = true;

        if (cornerDetectionFrame) {
            cancelAnimationFrame(
                cornerDetectionFrame
            );

            cornerDetectionFrame = null;
        }
    }


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

    if (automatic && !pageCornersDetected) {
        if (cornerDetectionStatus) {
            cornerDetectionStatus.textContent =
                "Align the OMR sheet Ã¢â‚¬â€ four black corner markers must be recognized before capture.";
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


    /*
        Keep the exact camera/upload preview visible after evaluation.

        The backend may perspective-correct and resize a COPY internally for
        recognition, but the user's captured preview must never be replaced
        by that canonical image. This avoids visible rotation, stretching,
        dimension changes, or tilt in the captured-image preview.
    */


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


async function scanBatchOMRs() {
    clearError();
    hideResult();
    hideSuccessState();

    const batchExam = (
        examSelect?.value
        || ""
    ).trim();

    if (
        !validateJeeScanMetadata(
            batchExam
        )
    ) {
        return true;
    }

    const files = batchUploadFiles.slice(
        0,
        MAX_BATCH_OMR_FILES
    );

    if (files.length <= 1) {
        return false;
    }

    showLoading(
        "Processing "
        + files.length
        + " OMR sheets..."
    );

    const formData = new FormData();

    for (const file of files) {
        formData.append(
            "images",
            file,
            file.name
        );
    }

    formData.append(
        "exam",
        examSelect.value
    );

    formData.append(
        "stream",
        selectedStream
    );

    appendJeeScanMetadata(
        formData,
        batchExam
    );

    try {
        const response = await fetch(
            "/scan-batch",
            {
                method: "POST",
                body: formData,
            }
        );

        const payload = await response.json();

        if (!response.ok) {
            throw new Error(
                payload.detail
                || "Batch OMR upload failed."
            );
        }

        for (const item of payload.results || []) {
            const result = item.result || {};
            const key = result.scan_id || result.id;

            if (key) {
                try {
                    localStorage.setItem(
                        `omr-result:${key}`,
                        JSON.stringify(result)
                    );
                } catch (storageError) {
                    console.debug(
                        "Result cache unavailable:",
                        storageError
                    );
                }
            }
        }

        const successful = payload.results || [];

        if (successful.length) {
            const lastResult = successful[successful.length - 1].result;

            latestResultId = (
                lastResult?.scan_id
                || lastResult?.id
                || null
            );

            displayResult(lastResult);
        }

        if (message) {
            message.textContent = (
                "Batch complete: "
                + payload.processed
                + " processed, "
                + payload.failed
                + " failed, out of "
                + payload.requested
                + "."
            );
        }

        batchUploadFiles = [];
        hideLoading();
        showSuccessState();

        return true;

    } catch (error) {
        hideLoading();

        showError(
            error.message
            || "Batch OMR upload failed."
        );

        return true;
    }
}

async function scanOMR() {
    if (batchUploadFiles.length > 1) {
        return scanBatchOMRs();
    }

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
        !validateJeeScanMetadata(
            exam
        )
    ) {
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
        "Ã¢Å¡Â¡ Scanning & Evaluating OMR Sheet... Analyzing Bubbles... Please Wait..."
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
            exam === "kcet" ? selectedStream : "pcm"
        );

        appendJeeScanMetadata(
            formData,
            exam
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

        // Prefer the durable database ID. The scan UUID remains a fallback for
        // local development and deployments without a configured database.
        latestResultId = data?.scan_id || data?.id || null;

        // Serverless local files may not survive the next request. Keep the
        // just-created result available to the individual-result page in this
        // browser even when no durable database is configured.
        try {
            [data?.id, data?.scan_id]
                .filter(Boolean)
                .forEach((resultKey) => {
                    localStorage.setItem(
                        `omr-result:${resultKey}`,
                        JSON.stringify(data)
                    );
                });
        } catch (storageError) {
            console.warn("Could not cache the OMR result locally:", storageError);
        }

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
        () => captureCameraImage(false)
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
    <h2>OMR Evaluation Report Ã¢â‚¬â€ Manchester Technologies</h2>
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

function localTodayIsoDate() {
    const now = new Date();
    now.setMinutes(
        now.getMinutes()
        - now.getTimezoneOffset()
    );
    return now.toISOString().slice(0, 10);
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

    if (jeeMetadataSection) {
        if (selected === "jee") {
            jeeMetadataSection.classList.remove("hidden");

            if (
                jeeExamDate
                && !jeeExamDate.value
            ) {
                jeeExamDate.value = localTodayIsoDate();
            }
        } else {
            jeeMetadataSection.classList.add("hidden");
        }
    }
}


function getJeeScanMetadata(examValue) {
    const exam = String(
        examValue || ""
    ).trim().toLowerCase();

    if (exam !== "jee") {
        return {
            valid: true,
            className: "",
            section: "",
            examDate: "",
            session: "",
        };
    }

    const className = (jeeClass?.value || "").trim();
    const section = (jeeSection?.value || "").trim();
    const examDate = (jeeExamDate?.value || "").trim();
    const session = (jeeSession?.value || "").trim();

    return {
        valid: Boolean(
            className
            && section
            && examDate
            && session
        ),
        className,
        section,
        examDate,
        session,
    };
}


function validateJeeScanMetadata(examValue) {
    const metadata = getJeeScanMetadata(
        examValue
    );

    if (!metadata.valid) {
        showError(
            "For JEE, select Class, Section, Exam Date and Session before scanning."
        );
        return null;
    }

    return metadata;
}


function appendJeeScanMetadata(
    formData,
    examValue
) {
    const metadata = getJeeScanMetadata(
        examValue
    );

    if (
        String(
            examValue || ""
        ).trim().toLowerCase()
        !== "jee"
    ) {
        return;
    }

    formData.append(
        "class_name",
        metadata.className
    );

    formData.append(
        "section",
        metadata.section
    );

    formData.append(
        "exam_date",
        metadata.examDate
    );

    formData.append(
        "session",
        metadata.session
    );
}

if (examSelect) {
    examSelect.addEventListener("change", updateExamStreamVisibility);
}

updateExamStreamVisibility();

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

