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

let selectedStream = "pcmb";


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


// Instant auto-capture when the sheet is fully visible.
const AUTO_CAPTURE_STABLE_CHECKS = 1;


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

    const maxMotion = Math.max(videoWidth, videoHeight) * 0.10;

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
        Check if corners are properly positioned.
        Stability check is done in monitorCornerBlocks.
    */

    if (!detection || !detection.sourcePoints) {
        return {
            ready: false,
            reason: "No corners detected"
        };
    }

    if (!isCompleteSheetInFrame(detection.sourcePoints, videoWidth, videoHeight)) {
        return {
            ready: false,
            reason: "Sheet partially outside frame"
        };
    }

    /* All positioning checks passed */
    return {
        ready: true,
        reason: "Positioned correctly"
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

        captureButton.disabled = false;
        captureButton.classList.remove("hidden");
    }
}


function cornerBlockMeasurement(
    pixels,
    width,
    height,
    startX,
    startY,
    endX,
    endY
) {

    let darkPixels = 0;

    let totalPixels = 0;

    let weightedX = 0;

    let weightedY = 0;

    /* Adaptive brightness threshold based on region lighting */
    let brightnessSum = 0;
    let pixelCount = 0;

    for (let y = startY; y < endY; y += 1) {

        for (let x = startX; x < endX; x += 1) {

            const offset = (y * width + x) * 4;

            const brightness =
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114;

            brightnessSum += brightness;
            pixelCount += 1;

            totalPixels += 1;
        }
    }

    /* Calculate average brightness and use as adaptive threshold */
    const avgBrightness = brightnessSum / Math.max(pixelCount, 1);
    const threshold = Math.min(avgBrightness * 0.5, 80);

    /* Second pass: find dark pixels using adaptive threshold */
    for (let y = startY; y < endY; y += 1) {

        for (let x = startX; x < endX; x += 1) {

            const offset = (y * width + x) * 4;

            const brightness =
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114;

            if (brightness < threshold) {

                darkPixels += 1;

                weightedX += x;

                weightedY += y;
            }
        }
    }

    return {
        coverage: darkPixels / Math.max(totalPixels, 1),
        x: darkPixels ? weightedX / darkPixels : 0,
        y: darkPixels ? weightedY / darkPixels : 0,
    };
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

    /* Even higher resolution for pixel-perfect accuracy */
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

    /* Optimized zones for better OMR corner detection */
    const zoneWidth = Math.round(analysisWidth * 0.25);

    const zoneHeight = Math.round(analysisHeight * 0.20);

    /* Positions slightly inset from edges for better accuracy */
    const zones = [
        [Math.round(analysisWidth * 0.02), Math.round(analysisHeight * 0.02)],
        [Math.round(analysisWidth * 0.73), Math.round(analysisHeight * 0.02)],
        [Math.round(analysisWidth * 0.02), Math.round(analysisHeight * 0.78)],
        [Math.round(analysisWidth * 0.73), Math.round(analysisHeight * 0.78)],
    ];

    const measurements = zones.map(
        ([x, y]) => cornerBlockMeasurement(
            pixels,
            analysisWidth,
            analysisHeight,
            x,
            y,
            Math.min(x + zoneWidth, analysisWidth),
            Math.min(y + zoneHeight, analysisHeight)
        )
    );

    if (!measurements.every(({ coverage }) => coverage >= 0.005)) {

        return null;
    }

    const displayWidth = cameraContainer.clientWidth;

    const displayHeight = cameraContainer.clientHeight;

    const displayPoints = measurements.map(({ x, y }) => ({
        x: (x / analysisWidth) * displayWidth,
        y: (y / analysisHeight) * displayHeight,
    }));

    const sourcePoints = measurements.map(({ x, y }) => ({
        x: (x / analysisWidth) * videoWidth,
        y: (y / analysisHeight) * videoHeight,
    }));

    /* Validate that corners are roughly in expected positions */
    const tl = sourcePoints[0];
    const tr = sourcePoints[1];
    const bl = sourcePoints[2];
    const br = sourcePoints[3];

    /* Check basic quad properties */
    const topEdgeDist = Math.abs(tl.y - tr.y);
    const bottomEdgeDist = Math.abs(bl.y - br.y);
    const leftEdgeDist = Math.abs(tl.x - bl.x);
    const rightEdgeDist = Math.abs(tr.x - br.x);

    /* All edges should be relatively parallel */
    if (Math.max(topEdgeDist, bottomEdgeDist) > videoHeight * 0.15) {
        return null;
    }
    if (Math.max(leftEdgeDist, rightEdgeDist) > videoWidth * 0.15) {
        return null;
    }

    return {
        displayPoints,
        sourcePoints,
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

        stableCornerChecks = detection
            ? stableCornerChecks + 1
            : 0;

        const videoWidth = camera.videoWidth;
        const videoHeight = camera.videoHeight;

        /*
            Check if positioned correctly for auto-capture.
        */
        const readinessCheck = isReadyForAutoCapture(
            detection,
            videoWidth,
            videoHeight
        );

        let statusMessage = null;

        if (readinessCheck.ready && !autoCaptureTriggered) {
            /*
                Sheet is correctly positioned. Capture immediately.
            */
            autoCaptureTriggered = true;
            setCornerDetectionState(
                true,
                "Capturing…"
            );

            /* Minimal delay for UI update */
            setTimeout(() => {
                captureCameraImage(true);
            }, 50);

            cornerDetectionFrame = requestAnimationFrame(
                monitorCornerBlocks
            );
            return;
        } else if (readinessCheck.ready) {
            /* Capture already triggered */
            statusMessage = "Capturing…";
        } else if (!readinessCheck.ready && detection) {
            /* Corners detected but not properly positioned */
            statusMessage = readinessCheck.reason;
        } else {
            /* No detection yet */
            statusMessage = null;
        }

        setCornerDetectionState(
            stableCornerChecks >= 1 && detection,
            statusMessage
        );

        drawDocumentBoundary(
            detection?.displayPoints,
            stableCornerChecks >= 1 && detection
        );

        if (stableCornerChecks >= 1 && detection) {

            detectedDocumentBounds = detection.sourcePoints;
        }

        /*
            Track previous frame for movement detection.
        */
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
            false;


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
    data
) {

    const result =
        data.result
        || data;


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
        bubbleDebugPreview.src =
            bubbleDebugUrl
            + "?t="
            + Date.now();

        if (bubbleAnalysisCard) {
            bubbleAnalysisCard.hidden = false;
            bubbleAnalysisCard.classList.remove("hidden");
        }
    }


    if (
        resultSection
    ) {

        resultSection.hidden =
            false;

        resultSection.classList.remove("hidden");

        resultSection.scrollIntoView({ behavior: "smooth" });
    }
}


/* ==========================================================
   SCAN
   ========================================================== */

async function scanOMR() {

    clearError();

    hideResult();


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


        displayResult(
            data
        );


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


            /*
                Allows selecting the same image again.
            */
            event.target.value =
                "";
        }
    );
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
