// static/app.js

console.log("OMR camera scanner loaded");


// ============================================================
// STATE
// ============================================================

let cameraStream = null;
let capturedBlob = null;


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
// BASIC UI HELPERS
// ============================================================

function showError(text) {

    if (!errorBox) {
        alert(text);
        return;
    }

    errorBox.textContent = text;

    errorBox.classList.remove(
        "hidden"
    );
}


function hideError() {

    if (!errorBox) {
        return;
    }

    errorBox.textContent = "";

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
// OPEN CAMERA
// ============================================================

async function openCamera() {

    hideError();
    hideResult();

    console.log(
        "Open Camera clicked"
    );


    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        showError(
            "Camera is not supported in this browser."
        );

        return;
    }


    try {

        // Stop old stream if one exists
        stopCamera();


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

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

                },

                audio: false

            });


        camera.srcObject =
            cameraStream;


        await camera.play();


        cameraContainer.classList.remove(
            "hidden"
        );


        captureButton.classList.remove(
            "hidden"
        );


        openCameraButton.classList.add(
            "hidden"
        );


        retakeButton.classList.add(
            "hidden"
        );


        scanButton.classList.add(
            "hidden"
        );


        preview.classList.add(
            "hidden"
        );


        capturedBlob = null;


        console.log(
            "Camera started successfully"
        );


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        let messageText =
            "Could not open camera.";


        if (
            error.name === "NotAllowedError"
        ) {

            messageText =
                "Camera permission was denied. Please allow camera access.";

        } else if (
            error.name === "NotFoundError"
        ) {

            messageText =
                "No camera was found on this device.";

        } else if (
            error.name === "NotReadableError"
        ) {

            messageText =
                "The camera is already being used by another application.";

        } else if (
            error.message
        ) {

            messageText =
                "Could not open camera: "
                + error.message;
        }


        showError(
            messageText
        );
    }
}


// ============================================================
// CAPTURE OMR
// ============================================================

function captureOMR() {

    hideError();


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


    // ========================================================
    // CAMERA FRAME SIZE
    // ========================================================

    const videoWidth =
        camera.videoWidth;

    const videoHeight =
        camera.videoHeight;


    console.log(
        "Camera resolution:",
        videoWidth,
        "x",
        videoHeight
    );


    // ========================================================
    // CROP REGION
    //
    // Must match the visual guide box:
    //
    // left   = 8%
    // top    = 5%
    // width  = 84%
    // height = 90%
    // ========================================================

    const cropX =
        videoWidth * 0.08;

    const cropY =
        videoHeight * 0.05;

    const cropWidth =
        videoWidth * 0.84;

    const cropHeight =
        videoHeight * 0.90;


    canvas.width =
        Math.round(
            cropWidth
        );

    canvas.height =
        Math.round(
            cropHeight
        );


    const context =
        canvas.getContext(
            "2d"
        );


    if (!context) {

        showError(
            "Could not create image capture canvas."
        );

        return;
    }


    // ========================================================
    // DRAW CROPPED CAMERA AREA
    // ========================================================

    context.drawImage(

        camera,

        cropX,
        cropY,
        cropWidth,
        cropHeight,

        0,
        0,
        canvas.width,
        canvas.height

    );


    // ========================================================
    // CONVERT TO JPEG
    // ========================================================

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


            // Revoke old preview URL if needed

            if (
                preview.src &&
                preview.src.startsWith(
                    "blob:"
                )
            ) {

                URL.revokeObjectURL(
                    preview.src
                );
            }


            preview.src =
                URL.createObjectURL(
                    blob
                );


            // Hide live camera
            cameraContainer.classList.add(
                "hidden"
            );


            captureButton.classList.add(
                "hidden"
            );


            // Show captured image
            preview.classList.remove(
                "hidden"
            );


            retakeButton.classList.remove(
                "hidden"
            );


            scanButton.classList.remove(
                "hidden"
            );


            console.log(
                "OMR captured:",
                blob.size,
                "bytes"
            );

        },

        "image/jpeg",

        0.95

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


    if (
        preview.src &&
        preview.src.startsWith(
            "blob:"
        )
    ) {

        URL.revokeObjectURL(
            preview.src
        );
    }


    preview.removeAttribute(
        "src"
    );


    preview.classList.add(
        "hidden"
    );


    retakeButton.classList.add(
        "hidden"
    );


    scanButton.classList.add(
        "hidden"
    );


    cameraContainer.classList.remove(
        "hidden"
    );


    captureButton.classList.remove(
        "hidden"
    );
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

            quality.innerHTML = `

                <h4>
                    Scan Quality
                </h4>

                <p>
                    Blur:
                    <strong>
                        ${q.blur ?? "-"}
                    </strong>
                </p>

                <p>
                    Brightness:
                    <strong>
                        ${q.brightness ?? "-"}
                    </strong>
                </p>

                <p>
                    Contrast:
                    <strong>
                        ${q.contrast ?? "-"}
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

            behavior: "smooth",

            block: "start"

        });
    }
}


// ============================================================
// SCAN OMR
// ============================================================

async function scanOMR() {

    hideError();
    hideResult();


    // ========================================================
    // EXAM
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
    // IMAGE
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
    // Backend receives only:
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
        "camera_omr.jpg"
    );


    // ========================================================
    // UI
    // ========================================================

    showLoading();


    if (scanButton) {

        scanButton.disabled =
            true;

        scanButton.textContent =
            "Scanning...";
    }


    // ========================================================
    // API REQUEST
    // ========================================================

    try {

        console.log(
            "Sending OMR to /scan..."
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


        let result;


        try {

            result =
                await response.json();

        } catch (jsonError) {

            throw new Error(
                "Server returned an invalid response."
            );
        }


        console.log(
            "Server response:",
            result
        );


        // ====================================================
        // API ERROR
        // ====================================================

        if (!response.ok) {

            throw new Error(
                result.detail
                ??
                result.message
                ??
                "OMR scan failed."
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
            "Scan error:",
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
// EVENT LISTENERS
// ============================================================

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


// ============================================================
// STOP CAMERA WHEN PAGE CLOSES
// ============================================================

window.addEventListener(
    "beforeunload",
    stopCamera
);


// ============================================================
// DEBUG
// ============================================================

console.log(
    "Exam element:",
    examSelect
);

console.log(
    "Open Camera button:",
    openCameraButton
);

console.log(
    "Camera element:",
    camera
);

console.log(
    "Capture button:",
    captureButton
);

console.log(
    "Scan button:",
    scanButton
);