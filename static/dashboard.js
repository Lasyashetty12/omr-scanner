"use strict";

const classFilter = document.getElementById("classFilter");
const sectionFilter = document.getElementById("sectionFilter");
const examDashboardFilter = document.getElementById("examDashboardFilter");
const dashboardSummary = document.getElementById("dashboardSummary");
const dashboardTableBody = document.getElementById("dashboardTableBody");

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatDate(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return escapeHtml(value);
    return date.toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
    });
}

function openIndividualResult(resultId) {
    if (!resultId) return;
    window.location.href = `/result.html?id=${encodeURIComponent(resultId)}`;
}

async function fetchAndRenderDashboard() {
    dashboardTableBody.innerHTML = `
        <tr>
            <td colspan="13" class="dashboard-message-cell">
                ⚡ Loading evaluated student records from database...
            </td>
        </tr>
    `;

    const params = new URLSearchParams();
    if (classFilter.value !== "all") params.set("class", classFilter.value);
    if (sectionFilter.value !== "all") params.set("section", sectionFilter.value);
    if (examDashboardFilter.value !== "all") params.set("exam", examDashboardFilter.value);

    try {
        const response = await fetch(`/api/omr-results?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const rows = await response.json();
        const results = Array.isArray(rows) ? rows : [];

        const totalStudents = results.length;
        const avgScore = totalStudents
            ? Math.round(results.reduce((sum, row) => sum + Number(row.score || 0), 0) / totalStudents)
            : 0;
        const passCount = results.filter((row) => Number(row.score || 0) >= 120).length;
        const goodCount = results.filter((row) => {
            const score = Number(row.score || 0);
            return score >= 90 && score < 120;
        }).length;

        dashboardSummary.innerHTML = `
            <div class="summary-card"><span>Total Students</span><strong>${totalStudents}</strong></div>
            <div class="summary-card"><span>Avg Score</span><strong>${avgScore}</strong></div>
            <div class="summary-card"><span>Pass</span><strong>${passCount}</strong></div>
            <div class="summary-card"><span>Good</span><strong>${goodCount}</strong></div>
        `;

        if (!results.length) {
            dashboardTableBody.innerHTML = `
                <tr><td colspan="13" class="dashboard-message-cell">No evaluated OMR results found.</td></tr>
            `;
            return;
        }

        dashboardTableBody.innerHTML = results.map((row) => `
            <tr>
                <td><strong>${escapeHtml(row.student_name || "Student Candidate")}</strong></td>
                <td>${escapeHtml(row.roll_number || "-")}</td>
                <td>${escapeHtml(row.class || "-")}</td>
                <td>${escapeHtml(row.section || "-")}</td>
                <td>${escapeHtml(row.exam || "-")}</td>
                <td>${escapeHtml(row.paper_code || "-")}</td>
                <td><strong>${escapeHtml(row.score ?? "-")}</strong></td>
                <td>${escapeHtml(row.correct ?? 0)}</td>
                <td>${escapeHtml(row.wrong ?? 0)}</td>
                <td>${escapeHtml(row.blank ?? 0)}</td>
                <td>${formatDate(row.exam_date || row.date)}</td>
                <td>${escapeHtml(row.session || "-")}</td>
                <td>
                    <button type="button" class="action-view-btn" data-result-id="${escapeHtml(row.scan_id || row.id)}">
                        View Result
                    </button>
                </td>
            </tr>
        `).join("");

        dashboardTableBody.querySelectorAll(".action-view-btn").forEach((button) => {
            button.addEventListener("click", () => {
                openIndividualResult(button.dataset.resultId);
            });
        });
    } catch (error) {
        console.error("Dashboard fetch error:", error);
        dashboardSummary.innerHTML = "";
        dashboardTableBody.innerHTML = `
            <tr>
                <td colspan="13" class="dashboard-message-cell dashboard-error-cell">
                    Unable to load student results: ${escapeHtml(error.message || "API failure")}
                </td>
            </tr>
        `;
    }
}

classFilter.addEventListener("change", fetchAndRenderDashboard);
sectionFilter.addEventListener("change", fetchAndRenderDashboard);
examDashboardFilter.addEventListener("change", fetchAndRenderDashboard);
document.addEventListener("DOMContentLoaded", fetchAndRenderDashboard);
