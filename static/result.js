"use strict";

const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const resultSection = document.getElementById("resultSection");

const resStudentName = document.getElementById("resStudentName");
const resRollNumber = document.getElementById("resRollNumber");
const resClassSection = document.getElementById("resClassSection");
const resExamDate = document.getElementById("resExamDate");
const resSession = document.getElementById("resSession");

const score = document.getElementById("score");
const resultExam = document.getElementById("resultExam");
const resultStream = document.getElementById("resultStream");
const paperCode = document.getElementById("paperCode");

const correct = document.getElementById("correct");
const wrong = document.getElementById("wrong");
const blank = document.getElementById("blank");
const multiple = document.getElementById("multiple");
const uncertain = document.getElementById("uncertain");

const quality = document.getElementById("quality");
const message = document.getElementById("message");

const questionTableBody = document.getElementById("questionTableBody");
const bubbleAnalysisCard = document.getElementById("bubbleAnalysisCard");
const bubbleDebugPreview = document.getElementById("bubbleDebugPreview");

const downloadPdfBtn = document.getElementById("downloadPdfBtn");
const downloadExcelBtn = document.getElementById("downloadExcelBtn");
const downloadWordBtn = document.getElementById("downloadWordBtn");

const imageLightbox = document.getElementById("imageLightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxClose = document.getElementById("lightboxClose");

// Vercel can route the result-page request to another instance immediately
// after /scan. Retry the durable Supabase lookup before showing "not found".
const RESULT_FETCH_DELAYS_MS = [
    0,
    250,
    500,
    900,
    1500,
    2500,
];

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchResultWithRetry(resultId) {
    let lastError = null;

    for (
        let attempt = 0;
        attempt < RESULT_FETCH_DELAYS_MS.length;
        attempt += 1
    ) {
        const delay = RESULT_FETCH_DELAYS_MS[attempt];

        if (delay > 0) {
            await sleep(delay);
        }

        try {
            const response = await fetch(
                `/api/omr-results/${encodeURIComponent(resultId)}?_t=${Date.now()}`,
                {
                    cache: "no-store",
                    headers: {
                        "Cache-Control": "no-cache",
                    },
                }
            );

            if (response.ok) {
                return await response.json();
            }

            lastError = new Error(
                `Result lookup returned HTTP ${response.status}`
            );
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error("Result not found.");
}


function showError(msg) {
    if (!errorBox) return;
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
    errorBox.hidden = false;
}

function hideLoading() {
    if (!loading) return;
    loading.classList.add("hidden");
    loading.hidden = true;
}

function readCachedResult(resultId) {
    try {
        const exact = (
            sessionStorage.getItem(`omr-result:${resultId}`)
            || localStorage.getItem(`omr-result:${resultId}`)
        );
        if (exact) return JSON.parse(exact);

        const latest = sessionStorage.getItem("omr-result:latest");
        if (!latest) return null;
        const parsed = JSON.parse(latest);
        return (
            String(parsed?.id ?? "") === String(resultId)
            || String(parsed?.scan_id ?? "") === String(resultId)
        ) ? parsed : null;
    } catch (storageError) {
        console.warn("Could not read cached OMR result:", storageError);
        return null;
    }
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

function displayResultData(data) {
    const result = data.result || data;

    const studentObj = result.student || {};
    const examInfoObj = result.exam_info || {};

    if (resStudentName) resStudentName.textContent = studentObj.name || result.student_name || "Student Candidate";
    if (resRollNumber) resRollNumber.textContent = studentObj.roll_number || result.roll_number || `ROLL-${result.id || result.scan_id || '101'}`;
    if (resClassSection) {
        const cls = studentObj.class || result.class || "12";
        const sec = studentObj.section || result.section || "A";
        resClassSection.textContent = `${cls} - Section ${sec}`;
    }
    if (resExamDate) {
        const rawDt = examInfoObj.exam_date || result.exam_date || result.date;
        resExamDate.textContent = rawDt ? new Date(rawDt).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) : "Today";
    }
    if (resSession) resSession.textContent = examInfoObj.session || result.session || "Morning";

    if (resultExam) resultExam.textContent = (result.exam || result.exam_name || "-").toUpperCase();
    if (resultStream) resultStream.textContent = (result.stream || "PCMB").toUpperCase();
    if (paperCode) paperCode.textContent = result.paper_code || result.series || result.jee_series || "-";

    if (score) score.textContent = result.score ?? "-";
    if (correct) correct.textContent = result.correct ?? 0;
    if (wrong) wrong.textContent = result.wrong ?? 0;
    if (blank) blank.textContent = result.blank ?? 0;
    if (multiple) multiple.textContent = result.multiple ?? 0;
    if (uncertain) uncertain.textContent = result.uncertain ?? 0;

    if (quality) {
        const qData = result.quality || data.quality;
        quality.textContent = qData ? `Blur: ${qData.blur ?? "-"} | Brightness: ${qData.brightness ?? "-"} | Contrast: ${qData.contrast ?? "-"}` : "";
    }

    if (message) message.textContent = result.message || data.message || "";

    renderQuestionTable(result.question_results);

    const bubbleDebugUrl =
        result.bubble_debug_image_url
        || data.bubble_debug_image_url
        || "";

    const bubbleDebugInline =
        result.bubble_debug_image_data_url
        || data.bubble_debug_image_data_url
        || "";

    if (bubbleDebugPreview) {
        let inlineFallbackUsed = false;

        const hideEvaluatedOmr = () => {
            bubbleDebugPreview.removeAttribute("src");
            if (bubbleAnalysisCard) {
                bubbleAnalysisCard.hidden = true;
                bubbleAnalysisCard.classList.add("hidden");
            }
        };

        const showEvaluatedOmr = () => {
            if (bubbleAnalysisCard) {
                bubbleAnalysisCard.hidden = false;
                bubbleAnalysisCard.classList.remove("hidden");
            }
        };

        bubbleDebugPreview.onload = showEvaluatedOmr;

        bubbleDebugPreview.onerror = () => {
            if (!inlineFallbackUsed && bubbleDebugInline) {
                inlineFallbackUsed = true;
                bubbleDebugPreview.src = bubbleDebugInline;
                return;
            }
            hideEvaluatedOmr();
        };

        if (bubbleDebugUrl) {
            bubbleDebugPreview.src = (
                bubbleDebugUrl.startsWith("data:")
                ? bubbleDebugUrl
                : bubbleDebugUrl + "?t=" + Date.now()
            );
        } else if (bubbleDebugInline) {
            inlineFallbackUsed = true;
            bubbleDebugPreview.src = bubbleDebugInline;
        } else {
            hideEvaluatedOmr();
        }
    }

    if (resultSection) {
        resultSection.hidden = false;
        resultSection.classList.remove("hidden");
    }
}

async function loadResult() {
    const urlParams = new URLSearchParams(window.location.search);
    const resultId = urlParams.get("id") || window.location.pathname.split("/").pop();

    if (!resultId || resultId === "result.html") {
        hideLoading();
        showError("Result not found. Invalid or missing result ID.");
        return;
    }

    // The /scan response is cached before navigation. Use it immediately.
    // This avoids "Result not found" on serverless deployments where the next
    // request may land on a different instance without the local JSON file.
    const immediateCached = readCachedResult(resultId);

    if (immediateCached) {
        hideLoading();
        displayResultData(immediateCached);
        return;
    }

    try {
        const data = await fetchResultWithRetry(resultId);
        try {
            localStorage.setItem(`omr-result:${resultId}`, JSON.stringify(data));
            sessionStorage.setItem(`omr-result:${resultId}`, JSON.stringify(data));
        } catch (storageError) {
            console.warn("Could not cache fetched OMR result:", storageError);
        }
        hideLoading();
        displayResultData(data);
    } catch (err) {
        console.error("Result fetch error:", err);
        const cached = readCachedResult(resultId);
        if (cached) {
            hideLoading();
            displayResultData(cached);
            return;
        }
        hideLoading();
        showError("Result not found.");
    }
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

if (downloadPdfBtn) downloadPdfBtn.addEventListener("click", downloadPDF);
if (downloadExcelBtn) downloadExcelBtn.addEventListener("click", downloadExcel);
if (downloadWordBtn) downloadWordBtn.addEventListener("click", downloadWord);

function openLightbox(src) {
    if (imageLightbox && lightboxImg && src) {
        lightboxImg.src = src;
        imageLightbox.classList.remove("hidden");
    }
}

function closeLightbox() {
    if (imageLightbox) imageLightbox.classList.add("hidden");
}

if (bubbleDebugPreview) {
    bubbleDebugPreview.addEventListener("click", () => openLightbox(bubbleDebugPreview.src));
}
if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
if (imageLightbox) {
    imageLightbox.addEventListener("click", (e) => {
        if (e.target === imageLightbox) closeLightbox();
    });
}

document.addEventListener("DOMContentLoaded", loadResult);
