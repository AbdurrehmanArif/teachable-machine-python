let videoStream = null;
let isTraining = false;
let isPredicting = false;
let predictInterval = null;
let recordInterval = null;

const videoElement = document.getElementById('inference-webcam');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// Initialize
async function init() {
    try {
        // Try with ideal constraints first, fallback to basic if needed
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 224 },
                    height: { ideal: 224 },
                    facingMode: "user"
                }
            });
        } catch (e) {
            console.log("Ideal constraints failed, trying basic...", e);
            videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        }

        videoElement.srcObject = videoStream;
        document.getElementById('status-indicator').textContent = "System Ready • Camera Active";
        document.getElementById('status-indicator').style.color = "var(--success)";

        // Set canvas size to match video
        videoElement.onloadedmetadata = () => {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
        };

        loadClasses();
    } catch (e) {
        console.error("Webcam error:", e);
        document.getElementById('status-indicator').textContent = "Camera Error: " + e.message;
        document.getElementById('status-indicator').style.color = "var(--error)";
        alert("Could not access webcam. Please check permissions and try again.");
    }
}

async function loadClasses() {
    const res = await fetch('/classes');
    const classes = await res.json();
    const container = document.getElementById('classes-list');
    container.innerHTML = '';

    classes.forEach(c => {
        const div = document.createElement('div');
        div.className = 'class-item';

        let imagesHtml = '';
        if (c.images && c.images.length > 0) {
            imagesHtml = '<div class="image-grid">';
            c.images.forEach(img => {
                imagesHtml += `<img src="/data/${c.name}/${img}" class="thumb-img" loading="lazy">`;
            });
            imagesHtml += '</div>';
        } else {
            imagesHtml = '<div class="no-images">No images yet</div>';
        }

        div.innerHTML = `
            <div class="class-header">
                <span class="class-name">${c.name}</span>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span id="count-${c.name}" style="font-size: 0.8rem; color: var(--text-secondary)">${c.count} images</span>
                    <button class="btn-icon" onclick="deleteClass('${c.name}')" title="Delete Class">×</button>
                </div>
            </div>
            ${imagesHtml}
            <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
                <button class="btn btn-sm" onmousedown="startRecording('${c.name}')" onmouseup="stopRecording()" onmouseleave="stopRecording()">Hold to Record</button>
                <label class="btn btn-sm" style="background: #334155; cursor: pointer; display: inline-block; margin: 0;">
                    Upload Files
                    <input type="file" accept="image/*" multiple onchange="uploadFiles('${c.name}', this)" style="display: none;">
                </label>
                <label class="btn btn-sm" style="background: #475569; cursor: pointer; display: inline-block; margin: 0;">
                    Upload Folder
                    <input type="file" webkitdirectory directory multiple onchange="uploadFiles('${c.name}', this)" style="display: none;">
                </label>
            </div>
        `;
        container.appendChild(div);
    });
}

async function addClass() {
    const nameInput = document.getElementById('new-class-name');
    const name = nameInput.value.trim();
    if (!name) return;

    await fetch(`/create_class/${name}`, { method: 'POST' });
    nameInput.value = '';
    loadClasses();
}

async function deleteClass(name) {
    if (!confirm(`Are you sure you want to delete class "${name}" and all its images?`)) return;

    await fetch(`/delete_class/${name}`, { method: 'DELETE' });
    loadClasses();
}

// Recording Logic
function startRecording(className) {
    if (recordInterval) clearInterval(recordInterval);
    captureAndUpload(className); // Immediate capture
    recordInterval = setInterval(() => captureAndUpload(className), 100); // Then every 100ms
}

function stopRecording() {
    if (recordInterval) {
        clearInterval(recordInterval);
        recordInterval = null;
        loadClasses(); // Update counts
    }
}

async function captureAndUpload(className) {
    if (!videoStream) return;

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        await fetch(`/upload/${className}`, {
            method: 'POST',
            body: formData
        });
    }, 'image/jpeg');
}

async function uploadFiles(className, input) {
    const files = input.files;
    if (!files || files.length === 0) return;

    const countLabel = document.getElementById(`count-${className}`);
    const originalText = countLabel.textContent;
    countLabel.textContent = "Uploading...";
    countLabel.style.color = "var(--primary)";

    let uploadedCount = 0;
    let totalFiles = files.length;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        // Skip non-image files
        if (!file.type.startsWith('image/')) {
            console.log(`Skipping non-image file: ${file.name}`);
            continue;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            await fetch(`/upload/${className}`, {
                method: 'POST',
                body: formData
            });
            uploadedCount++;

            // Update progress every 5 images
            if (uploadedCount % 5 === 0 || uploadedCount === totalFiles) {
                countLabel.textContent = `Uploaded ${uploadedCount}/${totalFiles}...`;
            }
        } catch (e) {
            console.error("Upload failed for", file.name, e);
        }
    }

    // Clear input and reload
    input.value = '';
    countLabel.textContent = "Refreshing...";
    await loadClasses();
}

// Training Logic
async function startTraining() {
    const btn = document.getElementById('train-btn');
    btn.disabled = true;

    document.getElementById('training-status-container').style.display = 'block';
    document.getElementById('results-container').style.display = 'none';

    const response = await fetch('/train', { method: 'POST' });
    const data = await response.json();

    if (data.error) {
        alert(data.message);
        btn.disabled = false;
        document.getElementById('training-status-container').style.display = 'none';
        return;
    }

    // Start polling
    const poll = setInterval(async () => {
        const res = await fetch('/state');
        const state = await res.json();

        const progressBar = document.getElementById('training-progress');
        const message = document.getElementById('training-message');

        progressBar.style.width = state.progress + '%';
        message.textContent = state.message;

        if (state.status === 'completed') {
            clearInterval(poll);
            btn.disabled = false;
            showResults(state.results);
        } else if (state.status === 'error') {
            clearInterval(poll);
            btn.disabled = false;
            message.style.color = 'var(--error)';
        }
    }, 1000);
}

function showResults(results) {
    document.getElementById('results-container').style.display = 'block';

    // Helper to format matrix
    const formatCM = (cm) => {
        if (!cm || cm.length === 0) return "No data";
        return cm.map(row => row.map(val => String(val).padStart(4)).join(" ")).join("\n");
    };

    if (results.logistic_regression) {
        document.getElementById('lr-acc').textContent = (results.logistic_regression.accuracy * 100).toFixed(1) + '%';
        const cmDiv = document.getElementById('lr-cm');
        cmDiv.style.display = 'block';
        cmDiv.textContent = "Confusion Matrix:\n" + formatCM(results.logistic_regression.confusion_matrix);
    }
    if (results.random_forest) {
        document.getElementById('rf-acc').textContent = (results.random_forest.accuracy * 100).toFixed(1) + '%';
        const cmDiv = document.getElementById('rf-cm');
        cmDiv.style.display = 'block';
        cmDiv.textContent = "Confusion Matrix:\n" + formatCM(results.random_forest.confusion_matrix);
    }
    if (results.cnn) {
        document.getElementById('cnn-acc').textContent = (results.cnn.accuracy * 100).toFixed(1) + '%';
        const cmDiv = document.getElementById('cnn-cm');
        cmDiv.style.display = 'block';
        cmDiv.textContent = "Confusion Matrix:\n" + formatCM(results.cnn.confusion_matrix);
    }
}

// Prediction Logic
function toggleLivePrediction() {
    const toggle = document.getElementById('live-predict-toggle');
    if (toggle.checked) {
        startPredictionLoop();
    } else {
        stopPredictionLoop();
    }
}

function startPredictionLoop() {
    if (predictInterval) clearInterval(predictInterval);
    predictInterval = setInterval(predict, 500); // Predict every 500ms
}

function stopPredictionLoop() {
    if (predictInterval) clearInterval(predictInterval);
}

async function predict() {
    if (!videoStream) return;

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'query.jpg');

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const predictions = await res.json();
            updatePredictionUI(predictions);
        } catch (e) {
            console.error("Prediction error", e);
        }
    }, 'image/jpeg');
}

function updatePredictionUI(predictions) {
    document.getElementById('pred-lr-val').textContent = predictions.logistic_regression || '--';
    document.getElementById('pred-rf-val').textContent = predictions.random_forest || '--';
    document.getElementById('pred-cnn-val').textContent = predictions.cnn || '--';

    // Highlight logic could go here if we had confidence scores
}

// Start
init();
