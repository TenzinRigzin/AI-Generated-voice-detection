document.addEventListener("DOMContentLoaded", () => {

    // API Configuration
    const API_URL = 'http://127.0.0.1:8000/predict';

    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const audioFileInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFile');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    const errorMessage = document.getElementById('errorMessage');
    const results = document.getElementById('results');
    const predictionValue = document.getElementById('predictionValue');
    const humanPercent = document.getElementById('humanPercent');
    const aiPercent = document.getElementById('aiPercent');
    const humanBar = document.getElementById('humanBar');
    const aiBar = document.getElementById('aiBar');

    let selectedFile = null;

    // Upload click
    uploadArea.addEventListener('click', () => {
        audioFileInput.click();
    });

    audioFileInput.addEventListener('change', (e) => {
        handleFileSelection(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFileSelection(e.dataTransfer.files[0]);
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showError('Please select an audio file first.');
            return;
        }
        await analyzeAudio();
    });

    function handleFileSelection(file) {
        if (!file) return;

        const validExtensions = ['wav', 'mp3'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(ext)) {
            showError('Upload WAV or MP3 only.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        uploadArea.style.display = 'none';
        fileInfo.style.display = 'flex';
        predictBtn.disabled = false;

        hideError();
        hideResults();
    }

    function clearFileSelection() {
        selectedFile = null;
        audioFileInput.value = '';
        uploadArea.style.display = 'block';
        fileInfo.style.display = 'none';
        predictBtn.disabled = true;
        hideError();
        hideResults();
    }

    function formatFileSize(bytes) {
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
    }

    async function analyzeAudio() {
        setLoadingState(true);
        hideError();
        hideResults();

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Backend error");
            }

            const data = await response.json();
            displayResults(data);

        } catch (err) {
            showError("Cannot connect to backend. Ensure FastAPI is running.");
            console.error(err);
        } finally {
            setLoadingState(false);
        }
    }

    function displayResults(data) {
        const human = Math.round(data.confidence.human * 100);
        const ai = Math.round(data.confidence.nonhuman * 100);

        predictionValue.textContent =
            data.prediction === 'human' ? 'Human Voice' : 'AI Generated';

        humanPercent.textContent = human + '%';
        aiPercent.textContent = ai + '%';

        humanBar.style.width = human + '%';
        aiBar.style.width = ai + '%';

        results.style.display = 'block';
    }

    function setLoadingState(state) {
        predictBtn.disabled = state;
        btnText.textContent = state ? 'Analyzing...' : 'Analyze Audio';
        spinner.style.display = state ? 'block' : 'none';
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    function hideResults() {
        results.style.display = 'none';
        humanBar.style.width = '0%';
        aiBar.style.width = '0%';
    }
});
