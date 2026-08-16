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


// Check runs four times per second.  Waiting two seconds avoids capturing
// while the phone is still moving into alignment.
const AUTO_CAPTURE_STABLE_CHECKS = 8;


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

    cameraContainer?.classList.remove(
        "page-corners-detected"
    );

    if (torchEnabled && cameraStream) {
        const track = cameraStream.getVideoTracks()[0];
        if (track) {
            try { track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) {}
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
   LIVE CORNER-BLOCK DETECTION
   ========================================================== */

function setCornerDetectionState(
    detected
) {

    pageCornersDetected = detected;

    cameraContainer?.classList.toggle(
        "page-corners-detected",
        detected
    );

    if (cornerDetectionStatus) {

        cornerDetectionStatus.textContent = detected
            ? "Page detected — all four corner blocks visible"
            : "Point camera at OMR sheet (Tap Capture anytime)";
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

    for (let y = startY; y < endY; y += 1) {

        for (let x = startX; x < endX; x += 1) {

            const offset = (y * width + x) * 4;

            const brightness =
                pixels[offset] * 0.299
                + pixels[offset + 1] * 0.587
                + pixels[offset + 2] * 0.114;

            if (brightness < 72) {

                darkPixels += 1;

                weightedX += x;

                weightedY += y;
            }

            totalPixels += 1;
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

    context.strokeStyle = detected ? "#31d57a" : "#ffd85a";

    context.shadowColor = "rgba(0, 0, 0, 0.75)";

    context.shadowBlur = 4;

    context.stroke();

    context.fillStyle = detected
        ? "rgba(49, 213, 122, 0.10)"
        : "rgba(255, 216, 90, 0.06)";

    context.fill();

    context.shadowBlur = 0;

    points.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, 5, 0, Math.PI * 2);
        context.fillStyle = detected ? "#31d57a" : "#ffd85a";
        context.fill();
    });
}


function detectDocumentCorners() {

    const videoWidth = camera.videoWidth;

    const videoHeight = camera.videoHeight;

    if (!videoWidth || !videoHeight) {

        return null;
    }

    const crop = calculateA4Crop(videoWidth, videoHeight);

    const analysisWidth = 180;

    const analysisHeight = Math.round(
        analysisWidth / A4_RATIO
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
        crop.x,
        crop.y,
        crop.width,
        crop.height,
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

    const zoneWidth = Math.round(analysisWidth * 0.18);

    const zoneHeight = Math.round(analysisHeight * 0.13);

    const zones = [
        [0, 0],
        [analysisWidth - zoneWidth, 0],
        [0, analysisHeight - zoneHeight],
        [analysisWidth - zoneWidth, analysisHeight - zoneHeight],
    ];

    const measurements = zones.map(
        ([x, y]) => cornerBlockMeasurement(
            pixels,
            analysisWidth,
            analysisHeight,
            x,
            y,
            x + zoneWidth,
            y + zoneHeight
        )
    );

    if (!measurements.every(({ coverage }) => coverage >= 0.015)) {

        return null;
    }

    const displayWidth = cameraContainer.clientWidth;

    const displayHeight = cameraContainer.clientHeight;

    const displayPoints = measurements.map(({ x, y }) => ({
        x: x / analysisWidth * displayWidth,
        y: y / analysisHeight * displayHeight,
    }));

    const sourcePoints = measurements.map(({ x, y }) => ({
        x: crop.x + x / analysisWidth * crop.width,
        y: crop.y + y / analysisHeight * crop.height,
    }));

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

        setCornerDetectionState(
            stableCornerChecks >= 2
        );

        drawDocumentBoundary(
            detection?.displayPoints,
            stableCornerChecks >= 2
        );

        if (stableCornerChecks >= 2) {

            detectedDocumentBounds = detection.sourcePoints;
        }
    }

    cornerDetectionFrame = requestAnimationFrame(
        monitorCornerBlocks
    );
}


function cropFromDetectedDocument(
    videoWidth,
    videoHeight
) {

    if (!detectedDocumentBounds?.length) {

        return calculateA4Crop(videoWidth, videoHeight);
    }

    const xs = detectedDocumentBounds.map(({ x }) => x);

    const ys = detectedDocumentBounds.map(({ y }) => y);

    const left = Math.min(...xs);

    const right = Math.max(...xs);

    const top = Math.min(...ys);

    const bottom = Math.max(...ys);

    // Provide a generous safety margin (6% of detected bounding dimensions)
    // to ensure corner registration marks are never clipped off.
    const marginX = (right - left) * 0.06;

    const marginY = (bottom - top) * 0.06;

    const x = Math.max(0, left - marginX);

    const y = Math.max(0, top - marginY);

    const maxRight = Math.min(videoWidth, right + marginX);

    const maxBottom = Math.min(videoHeight, bottom + marginY);

    return {
        x,
        y,
        width: Math.max(1, maxRight - x),
        height: Math.max(1, maxBottom - y),
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

    const sourceRatio =
        videoWidth
        / videoHeight;

    let cropWidth, cropHeight, cropX, cropY;

    if (
        sourceRatio
        > A4_RATIO
    ) {
        cropHeight = videoHeight;
        cropWidth = cropHeight * A4_RATIO;
        cropX = (videoWidth - cropWidth) / 2;
        cropY = 0;
    } else {
        cropWidth = videoWidth;
        cropHeight = cropWidth / A4_RATIO;
        cropX = 0;
        cropY = (videoHeight - cropHeight) / 2;
    }

    return {
        x: cropX,
        y: cropY,
        width: cropWidth,
        height: cropHeight
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

    const crop =
        cropFromDetectedDocument(
            videoWidth,
            videoHeight
        );

    captureCanvas.width = CAMERA_OUTPUT_WIDTH;
    captureCanvas.height = CAMERA_OUTPUT_HEIGHT;

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
        CAMERA_OUTPUT_WIDTH,
        CAMERA_OUTPUT_HEIGHT
    );

    context.drawImage(

        camera,

        crop.x,
        crop.y,
        crop.width,
        crop.height,

        0,
        0,

        CAMERA_OUTPUT_WIDTH,
        CAMERA_OUTPUT_HEIGHT
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
                : "uploaded_omr.jpg"
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
    streamPcmbBtn.addEventListener("click", function() {
        updateStreamUI("pcmb");
    });
}

if (streamPcmBtn) {
    streamPcmBtn.addEventListener("click", function() {
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
