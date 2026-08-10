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

    if (!loading) {
        return;
    }

    loading.classList.remove(
        "hidden"
    );
}


function hideLoading() {

    if (!loading) {
        return;
    }

    loading.classList.add(
        "hidden"
    );
}


function hideResult() {

    if (!resultSection) {
        return;
    }

    resultSection.classList.add(
        "hidden"
    );
}


function setButtonText(
    button,
    text
) {

    if (!button) {
        return;
    }

    button.textContent =
        text;
}


// ============================================================
// CAMERA SUPPORT CHECK
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

    console.log(
        "Open Camera clicked"
    );


    if (!examSelect) {

        showError(
            "Exam selector was not found."
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
            "Camera access is not supported in this browser. Use HTTPS or localhost."
        );

        return;
    }


    try {

        stopCamera();


        // ====================================================
        // Prefer rear camera.
        //
        // Do NOT request huge 4K frames.
        // Mobile devices can still provide good quality,
        // but this reduces browser and Vercel load.
        // ====================================================

        const constraints = {

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

        };


        cameraStream =
            await navigator.mediaDevices
                .getUserMedia(
                    constraints
                );


        if (!camera) {

            throw new Error(
                "Camera video element was not found."
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
            "Camera started:",
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
            "Could not open the camera.";


        if (
            error.name ===
            "NotAllowedError"
        ) {

            errorMessage =
                "Camera permission was denied. Allow camera access in your browser settings.";

        } else if (
            error.name ===
            "NotFoundError"
        ) {

            errorMessage =
                "No camera was found on this device.";

        } else if (
            error.name ===
            "NotReadableError"
        ) {

            errorMessage =
                "The camera could not be started. Another app may be using it.";

        } else if (
            error.name ===
            "OverconstrainedError"
        ) {

            errorMessage =
                "The requested camera configuration is not supported by this device.";

        } else if (
            error.name ===
            "SecurityError"
        ) {

            errorMessage =
                "Camera access was blocked for security reasons. Use the HTTPS site.";

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


    if (!camera) {

        showError(
            "Camera element was not found."
        );

        return;
    }


    if (
        !camera.videoWidth ||
        !camera.videoHeight
    ) {

        showError(
            "Camera is not ready yet. Wait a moment and try again."
        );

        return;
    }


    if (!canvas) {

        showError(
            "Capture canvas was not found."
        );

        return;
    }


    const sourceWidth =
        camera.videoWidth;

    const sourceHeight =
        camera.videoHeight;


    console.log(
        "Source camera resolution:",
        sourceWidth,
        "x",
        sourceHeight
    );


    // ========================================================
    // SEND FULL CAMERA FRAME
    //
    // The gold box is ONLY a visual guide.
    //
    // Python/OpenCV should detect the corner markers and
    // perform perspective correction.
    //
    // Do not crop according to CSS percentages.
    // ========================================================


    // ========================================================
    // RESIZE FOR VERCEL
    //
    // 1400 px wide is enough for OMR while avoiding huge
    // mobile images.
    // ========================================================

    const maxWidth =
        1400;


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
            "Could not initialize image capture."
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
                    "Could not capture the OMR image."
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
                "Captured JPEG size:",
                blob.size,
                "bytes"
            );


            // =================================================
            // PREVIEW URL
            // =================================================

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

        0.90

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


    // If the camera was stopped somehow,
    // reopen it.
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

function displayResult(
    result
) {

    console.log(
        "Displaying result:",
        result
    );


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


    // ========================================================
    // MESSAGE
    // ========================================================

    if (message) {

        message.textContent =
            result.message
            ?? "";
    }


    // ========================================================
    // SHOW RESULT
    // ========================================================

    if (resultSection) {

        resultSection.classList.remove(
            "hidden"
        );


        resultSection.scrollIntoView({

            behavior:
                "smooth",

            block:
                "start"

        });
    }
}


// ============================================================
// PARSE API RESPONSE
// ============================================================

async function parseServerResponse(
    response
) {

    const contentType =
        response.headers.get(
            "content-type"
        )
        || "";


    // ========================================================
    // JSON RESPONSE
    // ========================================================

    if (
        contentType.includes(
            "application/json"
        )
    ) {

        return await response.json();
    }


    // ========================================================
    // NON-JSON RESPONSE
    //
    // Vercel may return HTML for:
    // 500 / 502 / 504 / function crash / timeout
    // ========================================================

    let rawText =
        "";


    try {

        rawText =
            await response.text();

    } catch (error) {

        console.error(
            "Could not read server response:",
            error
        );
    }


    console.error(
        "NON-JSON SERVER RESPONSE"
    );


    console.error(
        "HTTP Status:",
        response.status
    );


    console.error(
        "Content-Type:",
        contentType
    );


    console.error(
        "Body:",
        rawText
    );


    let errorMessage =
        `Server error ${response.status}.`;


    if (
        response.status === 413
    ) {

        errorMessage =
            "Captured image is too large for the server.";

    } else if (
        response.status === 500
    ) {

        errorMessage =
            "Server error 500. The OMR backend crashed while processing the sheet.";

    } else if (
        response.status === 502
    ) {

        errorMessage =
            "Server error 502. The OMR processing function failed.";

    } else if (
        response.status === 504
    ) {

        errorMessage =
            "Server timeout 504. OMR processing took too long.";

    } else if (
        response.status === 404
    ) {

        errorMessage =
            "Scan API endpoint was not found.";

    } else if (
        response.status === 405
    ) {

        errorMessage =
            "The server does not allow this scan request.";
    }


    throw new Error(
        errorMessage
    );
}


// ============================================================
// SCAN OMR
// ============================================================

async function scanOMR() {

    hideError();
    hideResult();


    // ========================================================
    // VALIDATE EXAM
    // ========================================================

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


    // ========================================================
    // VALIDATE CAPTURE
    // ========================================================

    if (!capturedBlob) {

        showError(
            "Please capture the OMR sheet first."
        );

        return;
    }


    // ========================================================
    // FORM DATA
    //
    // No paper ID.
    // No answer-key selection.
    //
    // Backend detects paper code automatically.
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


    // ========================================================
    // UI
    // ========================================================

    showLoading();


    if (scanButton) {

        scanButton.disabled =
            true;

        setButtonText(
            scanButton,
            "Scanning..."
        );
    }


    // ========================================================
    // REQUEST
    // ========================================================

    try {

        console.log(
            "Uploading OMR..."
        );


        console.log(
            "Exam:",
            exam
        );


        console.log(
            "Image bytes:",
            capturedBlob.size
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


        console.log(
            "HTTP status:",
            response.status
        );


        const result =
            await parseServerResponse(
                response
            );


        console.log(
            "Server JSON:",
            result
        );


        // ====================================================
        // FASTAPI ERROR
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

            setButtonText(
                scanButton,
                "Scan & Evaluate"
            );
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
                function () {

                    resetScanner();
                }
            );

        } else {

            console.error(
                "exam element not found"
            );
        }


        console.log(
            "OMR scanner ready"
        );

    }
);


// ============================================================
// CLEANUP
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
// OPTIONAL GLOBAL ACCESS FOR DEBUGGING
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