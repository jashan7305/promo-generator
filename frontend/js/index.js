const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('videoFile');
const fileMeta = document.getElementById('fileMeta');
const sourcePreview = document.getElementById('sourcePreview');
const themeInput = document.getElementById('theme');
const submitButton = document.getElementById('submitButton');
const statusDiv = document.getElementById('status');
const progressBar = document.getElementById('progressBar');
const progressLabel = document.getElementById('progressLabel');
const pipelineState = document.getElementById('pipelineState');
const stageItems = [...document.querySelectorAll('#stageList li')];
const downloadSection = document.getElementById('downloadSection');
const promoVideo = document.getElementById('promoVideo');
const downloadLink = document.getElementById('downloadLink');
const clipList = document.getElementById('clipList');
const resultSummary = document.getElementById('resultSummary');
const resultDirection = document.getElementById('resultDirection');
const resultAnalysis = document.getElementById('resultAnalysis');
const resultDuration = document.getElementById('resultDuration');
const resultClips = document.getElementById('resultClips');
const resultTranscript = document.getElementById('resultTranscript');
const scriptLogline = document.getElementById('scriptLogline');
const scriptVoiceover = document.getElementById('scriptVoiceover');
const scriptTextCards = document.getElementById('scriptTextCards');
const scriptBeats = document.getElementById('scriptBeats');

const MAX_FILE_BYTES = 1024 * 1024 * 1024;
let sourcePreviewUrl = null;
let pollTimer = null;

function setStatus(message, progress = 0, state = 'STANDBY') {
    statusDiv.textContent = message;
    const percentage = Math.max(0, Math.min(100, Math.round(progress * 100)));
    progressBar.style.width = `${percentage}%`;
    progressLabel.textContent = `${percentage}%`;
    pipelineState.textContent = state;
}

function resetStages() {
    stageItems.forEach((item) => item.classList.remove('is-active', 'is-done'));
}

function updateStages(progress, stage) {
    const value = (stage || '').toLowerCase();
    const activeIndex = value.includes('transcrib') ? 1
        : value.includes('rank') ? 2
            : value.includes('cut') || value.includes('final') ? 3
                : value.includes('ready') ? 4 : 0;

    stageItems.forEach((item, index) => {
        item.classList.toggle('is-active', index === activeIndex && activeIndex < 4);
        item.classList.toggle('is-done', index < activeIndex);
    });
    setStatus(stage || 'Working locally…', progress, activeIndex === 4 ? 'COMPLETE' : 'PROCESSING');
}

function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setSelectedFile(file) {
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
        fileInput.value = '';
        fileMeta.textContent = 'That file is over the 1 GB limit.';
        setStatus('Choose a smaller video to continue.', 0, 'CHECK INPUT');
        return;
    }

    fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
    if (sourcePreviewUrl) URL.revokeObjectURL(sourcePreviewUrl);
    sourcePreviewUrl = URL.createObjectURL(file);
    sourcePreview.src = sourcePreviewUrl;
    sourcePreview.hidden = false;
    setStatus('Source selected. Add a theme and generate a promo.', 0, 'READY');
}

function showError(message) {
    setStatus(message, 0, 'CHECK INPUT');
    submitButton.disabled = false;
    submitButton.textContent = 'Try Again';
}

async function readJson(response) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Something went wrong. Try again.');
    return payload;
}

async function pollJob(statusUrl) {
    const job = await readJson(await fetch(statusUrl, { headers: { Accept: 'application/json' } }));
    updateStages(job.progress, job.stage);
    if (job.status === 'queued' || job.status === 'running') {
        pollTimer = window.setTimeout(() => pollJob(statusUrl), 1100);
        return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'The local cut failed.');
    renderResult(job.result);
}

function formatTime(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    return `${Math.floor(total / 60).toString().padStart(2, '0')}:${(total % 60).toString().padStart(2, '0')}`;
}

function renderResult(result) {
    downloadSection.hidden = false;
    promoVideo.pause();
    promoVideo.src = result.download_url;
    promoVideo.load();
    downloadLink.href = result.download_url;
    resultDuration.textContent = result.duration ?? '—';
    resultClips.textContent = result.clip_count ?? '—';
    resultTranscript.textContent = result.transcript_segments ?? '—';

    const formatLabel = result.aspect_ratio === 'vertical' ? '9:16' : result.aspect_ratio === 'square' ? '1:1' : 'source';
    resultDirection.textContent = `${(result.energy || 'balanced').replace('-', ' ')} · ${formatLabel}`.toUpperCase();
    resultAnalysis.textContent = `${result.has_dialogue ? 'Dialogue-aware' : 'Dialogue-free visual mode'} · ${result.analysis_provider || 'local analysis'}`;
    resultSummary.textContent = `A ${result.duration}-second cut shaped around “${result.theme}” using ${result.strategy === 'best-overall' ? 'the strongest overall moments' : result.strategy === 'story-arc' ? 'a full story arc' : result.strategy === 'visual-montage' ? 'dialogue and visual peaks' : 'theme relevance and story coverage'}.`;
    clipList.innerHTML = '';

    (result.clips || []).forEach((clip) => {
        const item = document.createElement('li');
        const time = document.createElement('span');
        const text = document.createElement('span');
        time.className = 'clip-time';
        text.className = 'clip-text';
        time.textContent = `${formatTime(clip.start)} — ${formatTime(clip.end)}`;
        text.textContent = clip.text;
        item.append(time, text);
        clipList.append(item);
    });

    const script = result.script || {};
    scriptLogline.textContent = `${script.title || 'Trailer outline'} — ${script.logline || ''}`;
    renderList(scriptVoiceover, script.voiceover || []);
    renderList(scriptTextCards, script.text_cards || []);
    renderList(scriptBeats, script.beats || []);

    downloadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    submitButton.disabled = false;
    submitButton.textContent = 'Generate Another Promo';
}

function renderList(target, values) {
    target.innerHTML = '';
    values.forEach((value) => {
        const item = document.createElement('li');
        item.textContent = value;
        target.append(item);
    });
}

fileInput.addEventListener('change', () => setSelectedFile(fileInput.files[0]));

document.querySelectorAll('.preset').forEach((button) => {
    button.addEventListener('click', () => {
        themeInput.value = button.dataset.theme;
        themeInput.focus();
    });
});

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (pollTimer) window.clearTimeout(pollTimer);
    resetStages();
    downloadSection.hidden = true;
    submitButton.disabled = true;
    submitButton.textContent = 'Generating…';

    const file = fileInput.files[0];
    const theme = themeInput.value.trim();
    if (!file) {
        showError('Please select a video file.');
        return;
    }
    if (theme.length < 2) {
        showError('Please enter a theme or story idea.');
        themeInput.focus();
        return;
    }

    try {
        updateStages(0.01, 'Uploading source');
        const uploadFormData = new FormData();
        uploadFormData.append('file', file);
        const uploadData = await readJson(await fetch('/api/upload', { method: 'POST', body: uploadFormData }));

        updateStages(0.08, 'Source validated');
        const generationForm = new FormData();
        generationForm.append('video_id', uploadData.video_id);
        generationForm.append('theme', theme);
        generationForm.append('target_duration', document.getElementById('targetDuration').value);
        generationForm.append('energy', document.getElementById('energy').value);
        generationForm.append('aspect_ratio', document.getElementById('aspectRatio').value);
        generationForm.append('strategy', document.getElementById('strategy').value);
        generationForm.append('analysis_mode', document.getElementById('analysisMode').value);
        const generationData = await readJson(await fetch('/api/generate', { method: 'POST', body: generationForm }));
        await pollJob(generationData.status_url);
    } catch (error) {
        showError(error.message || 'The local cut failed.');
    }
});

setStatus('Waiting for a video.', 0, 'STANDBY');
