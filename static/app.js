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


const quality =
    document.getElementById(
        "quality"
    );


const message =
    document.getElementById(
        "message"
    );


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


/* ==========================================================
   UI HELPERS
   ========================================================== */

function showError(
    text
) {

    if (!errorBox) {
        return;
    }


    errorBox.textContent =
        text;


    errorBox.hidden =
        false;
}


function clearError() {

    if (!errorBox) {
        return;
    }


    errorBox.textContent =
        "";


    errorBox.hidden =
        true;
}


function showLoading(
    text = "Processing OMR..."
) {

    if (!loading) {
        return;
    }


    loading.textContent =
        text;


    loading.hidden =
        false;
}


function hideLoading() {

    if (!loading) {
        return;
    }


    loading.hidden =
        true;
}


function hideResult() {

    if (!resultSection) {
        return;
    }


    resultSection.hidden =
        true;
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


        camera.srcObject =
            cameraStream;


        camera.hidden =
            false;


        capturedPreview.hidden =
            true;


        cameraContainer.hidden =
            false;


        captureButton.hidden =
            false;


        retakeButton.hidden =
            true;


        scanButton.disabled =
            true;


        await camera.play();


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


    let cropWidth;

    let cropHeight;

    let cropX;

    let cropY;


    /*
        Camera frame wider than A4:
        crop left/right.
    */
    if (
        sourceRatio
        > A4_RATIO
    ) {

        cropHeight =
            videoHeight;


        cropWidth =
            cropHeight
            * A4_RATIO;


        cropX =
            (
                videoWidth
                - cropWidth
            )
            / 2;


        cropY =
            0;
    }


    /*
        Camera frame taller/narrower than A4:
        crop top/bottom.
    */
    else {

        cropWidth =
            videoWidth;


        cropHeight =
            cropWidth
            / A4_RATIO;


        cropX =
            0;


        cropY =
            (
                videoHeight
                - cropHeight
            )
            / 2;
    }


    return {

        x:
            cropX,

        y:
            cropY,

        width:
            cropWidth,

        height:
            cropHeight
    };
}


/* ==========================================================
   CAPTURE CAMERA
   ========================================================== */

function captureCameraImage() {

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
        calculateA4Crop(
            videoWidth,
            videoHeight
        );


    captureCanvas.width =
        CAMERA_OUTPUT_WIDTH;


    captureCanvas.height =
        CAMERA_OUTPUT_HEIGHT;


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


    /*
        White background prevents transparent
        or undefined regions.
    */
    context.fillStyle =
        "#ffffff";


    context.fillRect(
        0,
        0,
        CAMERA_OUTPUT_WIDTH,
        CAMERA_OUTPUT_HEIGHT
    );


    /*
        Crop the center of the actual camera frame
        to exact A4 portrait ratio.
    */
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


            camera.hidden =
                true;


            captureButton.hidden =
                true;


            retakeButton.hidden =
                false;


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


    const objectUrl =
        URL.createObjectURL(
            file
        );


    const image =
        new Image();


    image.onload =
        function () {

            try {

                const sourceWidth =
                    image.naturalWidth;


                const sourceHeight =
                    image.naturalHeight;


                if (
                    !sourceWidth
                    ||
                    !sourceHeight
                ) {

                    throw new Error(
                        "Invalid uploaded image."
                    );
                }


                /*
                    Uploaded images should NOT be forcibly
                    cropped to A4 because some existing
                    calibrated files may already be correctly
                    prepared.

                    Resize only.
                */

                const maximumWidth =
                    1400;


                let outputWidth =
                    sourceWidth;


                let outputHeight =
                    sourceHeight;


                if (
                    outputWidth
                    > maximumWidth
                ) {

                    const scale =
                        maximumWidth
                        / outputWidth;


                    outputWidth =
                        Math.round(
                            outputWidth
                            * scale
                        );


                    outputHeight =
                        Math.round(
                            outputHeight
                            * scale
                        );
                }


                captureCanvas.width =
                    outputWidth;


                captureCanvas.height =
                    outputHeight;


                const context =
                    captureCanvas
                        .getContext(
                            "2d",
                            {
                                alpha:
                                    false
                            }
                        );


                context.fillStyle =
                    "#ffffff";


                context.fillRect(
                    0,
                    0,
                    outputWidth,
                    outputHeight
                );


                context.drawImage(
                    image,
                    0,
                    0,
                    outputWidth,
                    outputHeight
                );


                captureCanvas.toBlob(

                    function (
                        blob
                    ) {

                        URL.revokeObjectURL(
                            objectUrl
                        );


                        if (
                            !blob
                        ) {

                            showError(
                                "Could not prepare uploaded image."
                            );

                            return;
                        }


                        capturedBlob =
                            blob;


                        capturedFromCamera =
                            false;


                        clearPreviewUrl();


                        previewObjectUrl =
                            URL.createObjectURL(
                                blob
                            );


                        capturedPreview.src =
                            previewObjectUrl;


                        capturedPreview.hidden =
                            false;


                        camera.hidden =
                            true;


                        cameraContainer.hidden =
                            false;


                        captureButton.hidden =
                            true;


                        retakeButton.hidden =
                            false;


                        scanButton.disabled =
                            false;

                    },

                    "image/jpeg",

                    0.92
                );


            } catch (
            error
            ) {

                URL.revokeObjectURL(
                    objectUrl
                );


                console.error(
                    error
                );


                showError(
                    error.message
                    ||
                    "Unable to prepare uploaded image."
                );
            }
        };


    image.onerror =
        function () {

            URL.revokeObjectURL(
                objectUrl
            );


            showError(
                "Unable to read uploaded image."
            );
        };


    image.src =
        objectUrl;
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
            result.exam
            ||
            result.exam_name
            ||
            "-";
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


    if (
        resultSection
    ) {

        resultSection.hidden =
            false;
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


    showLoading(
        "Scanning OMR..."
    );


    try {

        const formData =
            new FormData();


        formData.append(
            "exam",
            exam
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