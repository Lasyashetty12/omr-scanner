// static/app.js

console.log("OMR scanner frontend loaded");


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
    document.getElementById(
        "exam"
    );

const openCameraButton =
    document.getElementById(
        "openCameraButton"
    );

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

const canvas =
    document.getElementById(
        "captureCanvas"
    );

const preview =
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


// ============================================================
// RESULT ELEMENTS
// ============================================================

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

const quality =
    document.getElementById(
        "quality"
    );

const message =
    document.getElementById(
        "message"
    );


// ============================================================
// UI HELPERS
// ============================================================

function showError(
    text
) {

    console.error(
        text
    );


    if (!errorBox) {

        alert(
            text
        );

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
// PREVIEW CLEANUP
// ============================================================

function clearPreviewUrl() {

    if (previewObjectUrl) {

        URL.revokeObjectURL(
            previewObjectUrl
        );

        previewObjectUrl =
            null;
    }
}


// ============================================================
// CAMERA SUPPORT
// ============================================================

function cameraSupported() {

    return Boolean(

        navigator.mediaDevices

        &&

        navigator.mediaDevices
            .getUserMedia

    );
}


// ============================================================
// OPEN CAMERA
// ============================================================

async function openCamera() {

    hideError();

    hideResult();


    if (
        !examSelect ||
        !examSelect.value
    ) {

        showError(
            "Please select an exam first."
        );

        return;
    }


    if (!cameraSupported()) {

        showError(
            "Camera is not supported. Use HTTPS or localhost."
        );

        return;
    }


    try {

        stopCamera();


        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({

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
                        }

                    }

                });


        if (!camera) {

            throw new Error(
                "Camera element was not found."
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


        capturedBlob =
            null;


        clearPreviewUrl();


        if (preview) {

            preview.classList.add(
                "hidden"
            );
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


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        let text =
            "Could not open camera.";


        if (
            error.name ===
            "NotAllowedError"
        ) {

            text =
                "Camera permission was denied. Allow camera permission in your browser.";

        } else if (
            error.name ===
            "NotFoundError"
        ) {

            text =
                "No camera was found on this device.";

        } else if (
            error.name ===
            "NotReadableError"
        ) {

            text =
                "Camera is currently being used by another app.";

        } else if (
            error.name ===
            "SecurityError"
        ) {

            text =
                "Camera access requires HTTPS.";

        } else if (
            error.message
        ) {

            text =
                error.message;
        }


        showError(
            text
        );
    }
}


// ============================================================
// PREPARE IMAGE BLOB
// ============================================================

function prepareImageFromSource(
    source,
    sourceWidth,
    sourceHeight,
    callback
) {

    if (!canvas) {

        showError(
            "Capture canvas was not found."
        );

        return;
    }


    // ========================================================
    // MAX WIDTH
    //
    // Keeps Vercel request reasonably small.
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
                alpha:
                    false
            }
        );


    if (!context) {

        showError(
            "Could not initialize image processor."
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

        source,

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

        function (
            blob
        ) {

            if (!blob) {

                showError(
                    "Could not prepare OMR image."
                );

                return;
            }


            callback(
                blob,
                outputWidth,
                outputHeight
            );

        },

        "image/jpeg",

        0.82

    );
}


// ============================================================
// SET CAPTURED IMAGE
// ============================================================

function setCapturedImage(
    blob,
    width,
    height
) {

    capturedBlob =
        blob;


    clearPreviewUrl();


    previewObjectUrl =
        URL.createObjectURL(
            blob
        );


    console.log(
        "Prepared OMR image:",
        width,
        "x",
        height
    );


    console.log(
        "Image bytes:",
        blob.size
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
}


// ============================================================
// CAPTURE CAMERA IMAGE
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


    const sourceWidth =
        camera.videoWidth;


    const sourceHeight =
        camera.videoHeight;


    console.log(
        "Capturing camera frame:",
        sourceWidth,
        "x",
        sourceHeight
    );


    // IMPORTANT:
    // Capture the whole camera frame.
    // The guide box is visual only.
    // OpenCV performs alignment.

    prepareImageFromSource(

        camera,

        sourceWidth,

        sourceHeight,

        function (
            blob,
            width,
            height
        ) {

            setCapturedImage(
                blob,
                width,
                height
            );
        }

    );
}


// ============================================================
// UPLOAD IMAGE
// ============================================================

function handleImageUpload(
    event
) {

    hideError();

    hideResult();


    if (
        !examSelect ||
        !examSelect.value
    ) {

        showError(
            "Please select an exam first."
        );


        event.target.value =
            "";


        return;
    }


    const file =
        event.target.files
            ? event.target.files[0]
            : null;


    if (!file) {

        return;
    }


    const allowedTypes = [

        "image/jpeg",

        "image/png"

    ];


    if (
        !allowedTypes.includes(
            file.type
        )
    ) {

        showError(
            "Only JPG, JPEG and PNG images are supported."
        );


        event.target.value =
            "";


        return;
    }


    console.log(
        "Selected file:",
        file.name
    );


    console.log(
        "Original upload bytes:",
        file.size
    );


    const img =
        new Image();


    const inputObjectUrl =
        URL.createObjectURL(
            file
        );


    img.onload =
        function () {


            console.log(
                "Uploaded image resolution:",
                img.naturalWidth,
                "x",
                img.naturalHeight
            );


            prepareImageFromSource(

                img,

                img.naturalWidth,

                img.naturalHeight,

                function (
                    blob,
                    width,
                    height
                ) {

                    URL.revokeObjectURL(
                        inputObjectUrl
                    );


                    stopCamera();


                    setCapturedImage(
                        blob,
                        width,
                        height
                    );
                }

            );
        };


    img.onerror =
        function () {

            URL.revokeObjectURL(
                inputObjectUrl
            );


            showError(
                "Could not read the uploaded image."
            );
        };


    img.src =
        inputObjectUrl;
}


// ============================================================
// RETAKE / RESET IMAGE
// ============================================================

function retakeOMR() {

    hideError();

    hideResult();


    capturedBlob =
        null;


    clearPreviewUrl();


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


    if (openCameraButton) {

        openCameraButton.classList.remove(
            "hidden"
        );
    }


    if (imageUpload) {

        imageUpload.value =
            "";
    }


    if (cameraStream) {

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

                typeof q.blur ===
                    "number"

                    ?

                    q.blur.toFixed(
                        2
                    )

                    :

                    q.blur ?? "-";


            const brightnessValue =

                typeof q.brightness ===
                    "number"

                    ?

                    q.brightness.toFixed(
                        2
                    )

                    :

                    q.brightness ?? "-";


            const contrastValue =

                typeof q.contrast ===
                    "number"

                    ?

                    q.contrast.toFixed(
                        2
                    )

                    :

                    q.contrast ?? "-";


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

            behavior:
                "smooth",

            block:
                "start"

        });
    }
}


// ============================================================
// SERVER RESPONSE PARSER
// ============================================================

async function parseServerResponse(
    response
) {

    const contentType =
        response.headers.get(
            "content-type"
        )
        || "";


    const rawText =
        await response.text();


    console.log(
        "HTTP status:",
        response.status
    );


    console.log(
        "Response content-type:",
        contentType
    );


    console.log(
        "Raw response:",
        rawText
    );


    if (!rawText) {

        throw new Error(
            `Server returned an empty response. HTTP ${response.status}`
        );
    }


    // ========================================================
    // JSON RESPONSE
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
                "JSON parsing error:",
                error
            );


            throw new Error(
                `Server returned invalid JSON. HTTP ${response.status}`
            );
        }
    }


    // ========================================================
    // NON-JSON RESPONSE
    // ========================================================

    console.error(
        "NON-JSON RESPONSE:",
        rawText
    );


    if (
        response.status ===
        413
    ) {

        throw new Error(
            "OMR image is too large for the server. HTTP 413."
        );
    }


    if (
        response.status ===
        500
    ) {

        throw new Error(
            "Backend crashed while processing OMR. HTTP 500."
        );
    }


    if (
        response.status ===
        502
    ) {

        throw new Error(
            "Vercel backend function failed. HTTP 502."
        );
    }


    if (
        response.status ===
        504
    ) {

        throw new Error(
            "OMR processing timed out. HTTP 504."
        );
    }


    if (
        response.status ===
        404
    ) {

        throw new Error(
            "Scan API endpoint was not found. HTTP 404."
        );
    }


    throw new Error(
        `Server returned a non-JSON response. HTTP ${response.status}.`
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

            ?

            examSelect.value

            :

            "";


    if (!exam) {

        showError(
            "Please select an exam."
        );

        return;
    }


    if (!capturedBlob) {

        showError(
            "Please capture or upload an OMR image first."
        );

        return;
    }


    // ========================================================
    // FORM DATA
    //
    // Only these are sent:
    //
    // exam
    // image
    //
    // Paper code / answer key is automatic.
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
        "omr_scan.jpg"
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
            "Sending OMR scan..."
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


        const result =
            await parseServerResponse(
                response
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


    clearPreviewUrl();


    stopCamera();


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


    if (imageUpload) {

        imageUpload.value =
            "";
    }
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


        // ====================================================
        // CAMERA
        // ====================================================

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


        // ====================================================
        // CAPTURE
        // ====================================================

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


        // ====================================================
        // UPLOAD
        // ====================================================

        if (imageUpload) {

            imageUpload.addEventListener(

                "change",

                handleImageUpload

            );

        } else {

            console.error(
                "imageUpload not found"
            );
        }


        // ====================================================
        // RETAKE
        // ====================================================

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


        // ====================================================
        // SCAN
        // ====================================================

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


        // ====================================================
        // EXAM CHANGE
        // ====================================================

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


        clearPreviewUrl();

    }

);


// ============================================================
// GLOBAL DEBUG ACCESS
// ============================================================

window.openCamera =
    openCamera;

window.captureOMR =
    captureOMR;

window.scanOMR =
    scanOMR;

window.retakeOMR =
    retakeOMR;

window.stopCamera =
    stopCamera;