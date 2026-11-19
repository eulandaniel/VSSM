document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("file-upload");
    const selectedFile = document.getElementById("selected-file");
    const dropzone = document.querySelector(".dropzone");

    if (!input || !selectedFile || !dropzone) {
        return;
    }

    input.addEventListener("change", () => {
        if (input.files.length > 0) {
            selectedFile.innerText = input.files[0].name + " selecionado";
            selectedFile.classList.add("active");
            dropzone.classList.add("has-file");
        } else {
            selectedFile.innerText = "Nenhum arquivo selecionado";
            selectedFile.classList.remove("active");
            dropzone.classList.remove("has-file");
        }
    });
});