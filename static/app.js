async function scanOMR() {

    const imageInput =
        document.getElementById(
            "imageInput"
        );

    const templateInput =
        document.getElementById(
            "templateInput"
        );

    const answerKey =
        document.getElementById(
            "answerKey"
        );

    const correctMarks =
        document.getElementById(
            "correctMarks"
        );

    const wrongMarks =
        document.getElementById(
            "wrongMarks"
        );

    const blankMarks =
        document.getElementById(
            "blankMarks"
        );

    const multipleMarks =
        document.getElementById(
            "multipleMarks"
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


    errorBox.classList.add(
        "hidden"
    );

    resultSection.classList.add(
        "hidden"
    );


    if (
        imageInput.files.length === 0
    ) {

        showError(
            "Please select an OMR image."
        );

        return;
    }


    if (
        answerKey.value.trim() === ""
    ) {

        showError(
            "Please enter the answer key."
        );

        return;
    }


    try {

        JSON.parse(
            answerKey.value
        );

    } catch {

        showError(
            "Answer key must be valid JSON."
        );

        return;
    }


    const formData =
        new FormData();


    formData.append(
        "image",
        imageInput.files[0]
    );


    formData.append(
        "template_name",
        templateInput.value
    );


    formData.append(
        "answer_key_json",
        answerKey.value
    );


    formData.append(
        "correct_marks",
        correctMarks.value
    );


    formData.append(
        "wrong_marks",
        wrongMarks.value
    );


    formData.append(
        "blank_marks",
        blankMarks.value
    );


    formData.append(
        "multiple_marks",
        multipleMarks.value
    );


    loading.classList.remove(
        "hidden"
    );


    try {

        const response =
            await fetch(
                "/scan",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(
                data.detail ||
                "OMR scanning failed."
            );

        }


        displayResult(
            data
        );


    } catch (error) {

        showError(
            error.message
        );

    } finally {

        loading.classList.add(
            "hidden"
        );

    }

}


function showError(message) {

    const errorBox =
        document.getElementById(
            "error"
        );

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

}


function displayResult(data) {

    document.getElementById(
        "score"
    ).textContent =
        data.score;


    document.getElementById(
        "correct"
    ).textContent =
        data.correct;


    document.getElementById(
        "wrong"
    ).textContent =
        data.wrong;


    document.getElementById(
        "blank"
    ).textContent =
        data.blank;


    document.getElementById(
        "multiple"
    ).textContent =
        data.multiple;


    const quality =
        document.getElementById(
            "quality"
        );


    quality.innerHTML = `
        Blur:
        <strong>
            ${data.quality.blur}
        </strong>

        &nbsp;&nbsp;

        Brightness:
        <strong>
            ${data.quality.brightness}
        </strong>

        &nbsp;&nbsp;

        Contrast:
        <strong>
            ${data.quality.contrast}
        </strong>
    `;


    const tableBody =
        document.getElementById(
            "resultTable"
        );


    tableBody.innerHTML = "";


    const results =
        data.question_results;


    Object.keys(
        results
    )
        .sort(
            (a, b) =>
                Number(a) - Number(b)
        )
        .forEach(
            question => {

                const item =
                    results[
                    question
                    ];


                const row =
                    document.createElement(
                        "tr"
                    );


                row.innerHTML = `
                <td>
                    ${question}
                </td>

                <td>
                    ${item.detected}
                </td>

                <td>
                    ${item.correct_answer}
                </td>

                <td
                    class="
                        status-${item.status}
                    "
                >
                    ${item.status}
                </td>

                <td>
                    ${item.marks}
                </td>
            `;


                tableBody.appendChild(
                    row
                );

            }
        );


    document.getElementById(
        "resultSection"
    )
        .classList.remove(
            "hidden"
        );


    document.getElementById(
        "resultSection"
    )
        .scrollIntoView({
            behavior: "smooth"
        });

}