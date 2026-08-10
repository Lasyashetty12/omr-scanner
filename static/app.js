// static/app.js

console.log("OMR camera scanner loaded");


// ============================================================
// STATE
// ============================================================

let cameraStream = null;
let capturedBlob = null;
let previewObjectUrl = null;


// ============================================================
// ELEMENTS
// ============================================================

const examSelect =
    document.getElementById("exam");

const openCameraButton =
    document.getElementById("openCameraButton");

const cameraContainer =
    document.getElementById("cameraContainer");

const camera =
    document.getElementById("camera");

const captureButton =
    document.getElementById("captureButton");

const retakeButton =
    document.getElementById("retakeButton");

const scanButton =
    document.getElementById("scanButton");

const canvas =
    document.getElementById("captureCanvas");

const preview =
    document.getElementById("capturedPreview");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("error");

const resultSection =
    document.getElementById("resultSection");


// ============================================================
// RESULT ELEMENTS
// ============================================================

const resultExam =
    document.getElementById("resultExam");

const paperCode =
    document.getElementById("paperCode");

const score =
    document.getElementById("score");

const correct =
    document.getElementById("correct");

const wrong =
    document.getElementById("wrong");

const blank =
    document.getElementById("blank");

const multiple =
    document.getElementById("multiple");

const quality =
    document.getElementById("quality");

const message =
    document.getElementById("message");


// ============================================================
// UI HELPERS
// ============================================================

function showError(text) {

    console.error(text);

    if (!errorBox) {

        alert(text);
        return;
    }

    errorBox.textContent =
        text;

    errorBox.classList.remove(
        "hidden"
    );
}


function hideError() {

    if (!errorBox) {
        return;
    }

    errorBox.textContent =
        "";

    errorBox.classList.add(
        "hidden"
    );
}


function showLoading() {

    if (loading) {

        loading.classList.remove(
            "hidden"
        );
    }
}


function hideLoading() {

    if (loading) {

        loading.classList.add(
            "hidden"
        );
    }
}


function hideResult() {

    if (resultSection) {

        resultSection.classList.add(
            "hidden"
        );
    }
}


// ============================================================
// CAMERA SUPPORT
// ============================================================

function cameraSupported() {

    return Boolean(
        navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia
    );
}


// ============================================================
// OPEN CAMERA
// ============================================================

async function openCamera() {

    hideError();
    hideResult();

    if (!examSelect) {

        showError(
            "Exam selector not found."
        );

        return;
    }


    if (!examSelect.value) {

        showError(
            "Please select an exam first."
        );

        return;
    }


    if (!cameraSupported()) {

        showError(
            "Camera is not supported in this browser. Use HTTPS."
        );

        return;
    }


    try {

        stopCamera();


        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({

                    audio: false,

                    video: {

                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 1920
                        },

                        height: {
                            ideal: 1080
                        }

                    }

                });


        if (!camera) {

            throw new Error(
                "Camera video element not found."
            );
        }


        camera.srcObject =
            cameraStream;


        camera.muted =
            true;


        camera.setAttribute(
            "playsinline",
            ""
        );


        await camera.play();


        console.log(
            "Camera resolution:",
            camera.videoWidth,
            "x",
            camera.videoHeight
        );


        if (cameraContainer) {

            cameraContainer.classList.remove(
                "hidden"
            );
        }


        if (captureButton) {

            captureButton.classList.remove(
                "hidden"
            );
        }


        if (openCameraButton) {

            openCameraButton.classList.add(
                "hidden"
            );
        }


        if (retakeButton) {

            retakeButton.classList.add(
                "hidden"
            );
        }


        if (scanButton) {

            scanButton.classList.add(
                "hidden"
            );
        }


        if (preview) {

            preview.classList.add(
                "hidden"
            );
        }


        capturedBlob =
            null;


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        let errorMessage =
            "Could not open camera.";


        if (
            error.name ===
            "NotAllowedError"
        ) {

            errorMessage =
                "Camera permission was denied.";

        } else if (
            error.name ===
            "NotFoundError"
        ) {

            errorMessage =
                "No camera was found.";

        } else if (
            error.name ===
            "NotReadableError"
        ) {

            errorMessage =
                "Camera is already being used by another app.";

        } else if (
            error.name ===
            "SecurityError"
        ) {

            errorMessage =
                "Camera access requires HTTPS.";

        } else if (
            error.message
        ) {

            errorMessage =
                error.message;
        }


        showError(
            errorMessage
        );
    }
}


// ============================================================
// CAPTURE OMR
// ============================================================

function captureOMR() {

    hideError();
    hideResult();


    if (
        !camera ||
        !camera.videoWidth ||
        !camera.videoHeight
    ) {

        showError(
            "Camera is not ready yet."
        );

        return;
    }


    if (!canvas) {

        showError(
            "Capture canvas not found."
        );

        return;
    }


    const sourceWidth =
        camera.videoWidth;

    const sourceHeight =
        camera.videoHeight;


    console.log(
        "Source frame:",
        sourceWidth,
        "x",
        sourceHeight
    );


    // ========================================================
    // FULL FRAME CAPTURE
    //
    // We do NOT crop by the visual guide box.
    // OpenCV performs actual marker detection and alignment.
    // ========================================================


    // ========================================================
    // RESIZE FOR VERCEL
    // ========================================================

    const maxWidth =
        1100;


    let outputWidth =
        sourceWidth;

    let outputHeight =
        sourceHeight;


    if (
        outputWidth >
        maxWidth
    ) {

        const resizeScale =
            maxWidth /
            outputWidth;


        outputWidth =
            Math.round(
                outputWidth *
                resizeScale
            );


        outputHeight =
            Math.round(
                outputHeight *
                resizeScale
            );
    }


    canvas.width =
        outputWidth;

    canvas.height =
        outputHeight;


    const context =
        canvas.getContext(
            "2d",
            {
                alpha: false
            }
        );


    if (!context) {

        showError(
            "Could not initialize capture canvas."
        );

        return;
    }


    context.clearRect(
        0,
        0,
        outputWidth,
        outputHeight
    );


    context.drawImage(

        camera,

        0,
        0,
        sourceWidth,
        sourceHeight,

        0,
        0,
        outputWidth,
        outputHeight

    );


    canvas.toBlob(

        function (blob) {

            if (!blob) {

                showError(
                    "Could not capture OMR image."
                );

                return;
            }


            capturedBlob =
                blob;


            console.log(
                "Captured image:",
                outputWidth,
                "x",
                outputHeight
            );


            console.log(
                "Captured bytes:",
                blob.size
            );


            if (previewObjectUrl) {

                URL.revokeObjectURL(
                    previewObjectUrl
                );
            }


            previewObjectUrl =
                URL.createObjectURL(
                    blob
                );


            if (preview) {

                preview.src =
                    previewObjectUrl;

                preview.classList.remove(
                    "hidden"
                );
            }


            if (cameraContainer) {

                cameraContainer.classList.add(
                    "hidden"
                );
            }


            if (captureButton) {

                captureButton.classList.add(
                    "hidden"
                );
            }


            if (retakeButton) {

                retakeButton.classList.remove(
                    "hidden"
                );
            }


            if (scanButton) {

                scanButton.classList.remove(
                    "hidden"
                );
            }

        },

        "image/jpeg",

        0.82

    );
}


// ============================================================
// RETAKE
// ============================================================

function retakeOMR() {

    hideError();
    hideResult();


    capturedBlob =
        null;


    if (previewObjectUrl) {

        URL.revokeObjectURL(
            previewObjectUrl
        );

        previewObjectUrl =
            null;
    }


    if (preview) {

        preview.removeAttribute(
            "src"
        );

        preview.classList.add(
            "hidden"
        );
    }


    if (retakeButton) {

        retakeButton.classList.add(
            "hidden"
        );
    }


    if (scanButton) {

        scanButton.classList.add(
            "hidden"
        );
    }


    if (!cameraStream) {

        openCamera();
        return;
    }


    if (cameraContainer) {

        cameraContainer.classList.remove(
            "hidden"
        );
    }


    if (captureButton) {

        captureButton.classList.remove(
            "hidden"
        );
    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(result) {

    if (resultExam) {

        resultExam.textContent =
            result.exam
            ?? "-";
    }


    if (paperCode) {

        paperCode.textContent =
            result.paper_code
            ??
            result.series
            ??
            "-";
    }


    if (score) {

        score.textContent =
            result.score
            ?? "-";
    }


    if (correct) {

        correct.textContent =
            result.correct
            ?? "-";
    }


    if (wrong) {

        wrong.textContent =
            result.wrong
            ?? "-";
    }


    if (blank) {

        blank.textContent =
            result.blank
            ?? "-";
    }


    if (multiple) {

        multiple.textContent =
            result.multiple
            ?? "-";
    }


    // ========================================================
    // QUALITY
    // ========================================================

    if (quality) {

        const q =
            result.quality;


        if (q) {

            const blurValue =
                typeof q.blur === "number"
                    ? q.blur.toFixed(2)
                    : q.blur ?? "-";


            const brightnessValue =
                typeof q.brightness === "number"
                    ? q.brightness.toFixed(2)
                    : q.brightness ?? "-";


            const contrastValue =
                typeof q.contrast === "number"
                    ? q.contrast.toFixed(2)
                    : q.contrast ?? "-";


            quality.innerHTML = `

                <h4>
                    Scan Quality
                </h4>

                <p>
                    Blur:
                    <strong>
                        ${blurValue}
                    </strong>
                </p>

                <p>
                    Brightness:
                    <strong>
                        ${brightnessValue}
                    </strong>
                </p>

                <p>
                    Contrast:
                    <strong>
                        ${contrastValue}
                    </strong>
                </p>
            `;

        } else {

            quality.innerHTML =
                "";
        }
    }


    if (message) {

        message.textContent =
            result.message
            ?? "";
    }


    if (resultSection) {

        resultSection.classList.remove(
            "hidden"
        );


        resultSection.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });
    }
}


// ============================================================
// PARSE SERVER RESPONSE
// ============================================================

async function parseServerResponse(
    response
) {

    const contentType =
        response.headers.get(
            "content-type"
        ) || "";


    const rawText =
        await response.text();


    console.log(
        "HTTP status:",
        response.status
    );


    console.log(
        "Content-Type:",
        contentType
    );


    console.log(
        "Raw response:",
        rawText
    );


    // ========================================================
    // EMPTY SERVER RESPONSE
    // ========================================================

    if (!rawText) {

        throw new Error(
            `Server returned an empty response. HTTP ${response.status}`
        );
    }


    // ========================================================
    // JSON
    // ========================================================

    if (
        contentType.includes(
            "application/json"
        )
    ) {

        try {

            return JSON.parse(
                rawText
            );

        } catch (error) {

            console.error(
                "JSON parse error:",
                error
            );


            throw new Error(
                `Server returned broken JSON. HTTP ${response.status}`
            );
        }
    }


    // ========================================================
    // NON JSON
    // ========================================================

    console.error(
        "NON-JSON RESPONSE:",
        rawText
    );


    if (
        response.status === 413
    ) {

        throw new Error(
            "Camera image is too large. HTTP 413."
        );
    }


    if (
        response.status === 500
    ) {

        throw new Error(
            "Backend crashed while processing OMR. HTTP 500."
        );
    }


    if (
        response.status === 502
    ) {

        throw new Error(
            "Vercel function failed. HTTP 502."
        );
    }


    if (
        response.status === 504
    ) {

        throw new Error(
            "OMR processing timed out. HTTP 504."
        );
    }


    if (
        response.status === 404
    ) {

        throw new Error(
            "Scan API endpoint not found. HTTP 404."
        );
    }


    throw new Error(
        `Server returned a non-JSON response. HTTP ${response.status}`
    );
}


// ============================================================
// SCAN OMR
// ============================================================

async function scanOMR() {

    hideError();
    hideResult();


    const exam =
        examSelect
            ? examSelect.value
            : "";


    if (!exam) {

        showError(
            "Please select an exam."
        );

        return;
    }


    if (!capturedBlob) {

        showError(
            "Please capture the OMR sheet first."
        );

        return;
    }


    // ========================================================
    // FORM DATA
    //
    // Only exam and image are sent.
    // Paper code is detected automatically.
    // ========================================================

    const formData =
        new FormData();


    formData.append(
        "exam",
        exam
    );


    formData.append(
        "image",
        capturedBlob,
        "camera_omr.jpg"
    );


    showLoading();


    if (scanButton) {

        scanButton.disabled =
            true;

        scanButton.textContent =
            "Scanning...";
    }


    try {

        console.log(
            "Sending scan request..."
        );


        console.log(
            "Exam:",
            exam
        );


        console.log(
            "Image size:",
            capturedBlob.size
        );


        const response =
            await fetch(
                "/scan",
                {

                    method: "POST",

                    body: formData

                }
            );


        const result =
            await parseServerResponse(
                response
            );


        // ====================================================
        // FASTAPI JSON ERROR
        // ====================================================

        if (!response.ok) {

            let errorMessage =
                "OMR scan failed.";


            if (
                result &&
                result.detail
            ) {

                if (
                    typeof result.detail ===
                    "string"
                ) {

                    errorMessage =
                        result.detail;

                } else {

                    errorMessage =
                        JSON.stringify(
                            result.detail
                        );
                }

            } else if (
                result &&
                result.message
            ) {

                errorMessage =
                    result.message;
            }


            throw new Error(
                errorMessage
            );
        }


        // ====================================================
        // SUCCESS
        // ====================================================

        console.log(
            "Scan result:",
            result
        );


        displayResult(
            result
        );


        stopCamera();


    } catch (error) {

        console.error(
            "Scan failed:",
            error
        );


        showError(
            error.message
            ||
            "OMR scan failed."
        );


    } finally {

        hideLoading();


        if (scanButton) {

            scanButton.disabled =
                false;

            scanButton.textContent =
                "Scan & Evaluate";
        }
    }
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    if (!cameraStream) {

        return;
    }


    const tracks =
        cameraStream.getTracks();


    for (
        const track of tracks
    ) {

        track.stop();
    }


    cameraStream =
        null;


    if (camera) {

        camera.srcObject =
            null;
    }


    console.log(
        "Camera stopped"
    );
}


// ============================================================
// RESET SCANNER
// ============================================================

function resetScanner() {

    hideError();
    hideResult();
    hideLoading();


    capturedBlob =
        null;


    if (previewObjectUrl) {

        URL.revokeObjectURL(
            previewObjectUrl
        );

        previewObjectUrl =
            null;
    }


    if (preview) {

        preview.removeAttribute(
            "src"
        );

        preview.classList.add(
            "hidden"
        );
    }


    if (cameraContainer) {

        cameraContainer.classList.add(
            "hidden"
        );
    }


    if (captureButton) {

        captureButton.classList.add(
            "hidden"
        );
    }


    if (retakeButton) {

        retakeButton.classList.add(
            "hidden"
        );
    }


    if (scanButton) {

        scanButton.classList.add(
            "hidden"
        );
    }


    if (openCameraButton) {

        openCameraButton.classList.remove(
            "hidden"
        );
    }


    stopCamera();
}


// ============================================================
// EVENT LISTENERS
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "DOM loaded"
        );


        if (openCameraButton) {

            openCameraButton.addEventListener(
                "click",
                openCamera
            );

        } else {

            console.error(
                "openCameraButton not found"
            );
        }


        if (captureButton) {

            captureButton.addEventListener(
                "click",
                captureOMR
            );

        } else {

            console.error(
                "captureButton not found"
            );
        }


        if (retakeButton) {

            retakeButton.addEventListener(
                "click",
                retakeOMR
            );

        } else {

            console.error(
                "retakeButton not found"
            );
        }


        if (scanButton) {

            scanButton.addEventListener(
                "click",
                scanOMR
            );

        } else {

            console.error(
                "scanButton not found"
            );
        }


        if (examSelect) {

            examSelect.addEventListener(
                "change",
                resetScanner
            );

        } else {

            console.error(
                "exam selector not found"
            );
        }


        console.log(
            "OMR scanner ready"
        );
    }
);


// ============================================================
// PAGE CLEANUP
// ============================================================

window.addEventListener(
    "beforeunload",
    function () {

        stopCamera();


        if (previewObjectUrl) {

            URL.revokeObjectURL(
                previewObjectUrl
            );
        }
    }
);


// ============================================================
// GLOBAL DEBUG ACCESS
// ============================================================

window.openCamera =
    openCamera;

window.captureOMR =
    captureOMR;

window.retakeOMR =
    retakeOMR;

window.scanOMR =
    scanOMR;

window.stopCamera =
    stopCamera;