const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('videoFile');
const musicInput = document.getElementById('musicFile');
const audioModeInput = document.getElementById('audioMode');
const analysisModeInput = document.getElementById('analysisMode');
const providerSelectors = document.getElementById('providerSelectors');
const transcriptionProviderInput = document.getElementById('transcriptionProvider');
const visualProviderInput = document.getElementById('visualProvider');
const elevenLabsKeyInput = document.getElementById('elevenLabsKey');
const geminiKeyInput = document.getElementById('geminiKey');
const elevenLabsCredentials = document.getElementById('elevenLabsCredentials');
const geminiCredentials = document.getElementById('geminiCredentials');
const fileMeta = document.getElementById('fileMeta');
const musicMeta = document.getElementById('musicMeta');
const sourcePreview = document.getElementById('sourcePreview');
const youtubeUrl = document.getElementById('youtubeUrl');
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
const agentSummary = document.getElementById('agentSummary');
const agentTrace = document.getElementById('agentTrace');

const MAX_FILE_BYTES = 1024 * 1024 * 1024;
const MAX_MUSIC_BYTES = 512 * 1024 * 1024;

function updateProviderFields() {
    const analysisMode = analysisModeInput.value;
    const transcriptionProvider = transcriptionProviderInput.value;
    const visualProvider = visualProviderInput.value;

    // Show/hide provider selectors based on analysis mode
    const isCustomProviders = analysisMode === 'providers';
    providerSelectors.hidden = !isCustomProviders;

    // Set defaults based on analysis mode
    if (analysisMode === 'local') {
        transcriptionProviderInput.value = 'local';
        visualProviderInput.value = 'local';
    } else if (analysisMode === 'gemini') {
        transcriptionProviderInput.value = 'local';
        visualProviderInput.value = 'gemini';
    } else if (analysisMode === 'elevenlabs') {
        transcriptionProviderInput.value = 'elevenlabs';
        visualProviderInput.value = 'local';
    }

    // Show/hide credentials based on selected providers
    const effectiveTranscription = isCustomProviders ? transcriptionProvider : transcriptionProviderInput.value;
    const effectiveVisual = isCustomProviders ? visualProvider : visualProviderInput.value;

    elevenLabsCredentials.hidden = effectiveTranscription !== 'elevenlabs';
    geminiCredentials.hidden = effectiveVisual !== 'gemini';
    elevenLabsKeyInput.required = effectiveTranscription === 'elevenlabs';
    geminiKeyInput.required = effectiveVisual === 'gemini';
    if (effectiveTranscription !== 'elevenlabs') elevenLabsKeyInput.value = '';
    if (effectiveVisual !== 'gemini') geminiKeyInput.value = '';
}
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
            : value.includes('soundtrack') || value.includes('music') ? 3
                : value.includes('cut') || value.includes('final') ? 4
                    : value.includes('ready') ? 5 : 0;

    stageItems.forEach((item, index) => {
        item.classList.toggle('is-active', index === activeIndex && activeIndex < 5);
        item.classList.toggle('is-done', index < activeIndex);
    });
    setStatus(stage || 'Working locally…', progress, activeIndex === 5 ? 'COMPLETE' : 'PROCESSING');
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

function setSelectedMusic(file) {
    if (!file) return;
    if (file.size > MAX_MUSIC_BYTES) {
        musicInput.value = '';
        musicMeta.textContent = 'That soundtrack is over the 512 MB limit.';
        setStatus('Choose a smaller soundtrack to continue.', 0, 'CHECK INPUT');
        return;
    }
    const mode = audioModeInput.value;
    musicMeta.textContent = `${file.name} · ${formatBytes(file.size)} · ${mode === 'music' ? 'music only' : mode === 'dialogue' ? 'low under dialogue' : 'mixed with source audio'}`;
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
    if (job.status === 'failed') throw new Error(job.error || 'The local cut failed.');
    updateStages(job.progress, job.stage);
    if (job.status === 'queued' || job.status === 'running') {
        pollTimer = window.setTimeout(() => pollJob(statusUrl), 1100);
        return;
    }
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
    const excerpt = result.music_excerpt;
    const excerptLabel = excerpt ? ` · track excerpt ${formatTime(excerpt.start)}–${formatTime(excerpt.end)}` : '';
    const tempoLabel = result.music_tempo_bpm && result.music_tempo_confidence >= 0.35
        ? ` · approx. ${Math.round(result.music_tempo_bpm)} BPM` : '';
    const duckLabel = result.music_duck_intervals ? ` · ducked under ${result.music_duck_intervals} dialogue passage(s)` : '';
    resultAnalysis.textContent = `${result.has_dialogue ? 'Dialogue-aware' : 'Dialogue-free visual mode'} · ${result.analysis_provider || 'local analysis'}${result.music_used ? ` · ${result.music_beat_count || 0} detected accents${excerptLabel}${tempoLabel}${duckLabel}` : ''}`;
    resultSummary.textContent = `A ${result.duration}-second cut shaped around “${result.theme}” using ${result.strategy === 'best-overall' ? 'the strongest overall moments' : result.strategy === 'story-arc' ? 'a full story arc' : result.strategy === 'visual-montage' ? 'dialogue and visual peaks' : 'theme relevance and story coverage'}.`;
    clipList.innerHTML = '';

    (result.clips || []).forEach((clip) => {
        const item = document.createElement('li');
        const time = document.createElement('span');
        const text = document.createElement('span');
        time.className = 'clip-time';
        text.className = 'clip-text';
        time.textContent = `Source ${formatTime(clip.source_start ?? clip.start)} — ${formatTime(clip.source_end ?? clip.end)} · Output ${formatTime(clip.output_start ?? 0)} — ${formatTime(clip.output_end ?? 0)}`;
        const speakers = clip.speakers?.length ? `Speakers ${clip.speakers.join(', ')}` : '';
        const mutedDialogue = (clip.dialogue || [])
            .filter((line) => line.audible === false)
            .map((line) => `Muted at a cut to avoid partial dialogue: ${line.text}`);
        text.textContent = [clip.text, speakers, clip.rationale, ...mutedDialogue].filter(Boolean).join(' · ');
        item.append(time, text);
        clipList.append(item);
    });

    const script = result.script || {};
    scriptLogline.textContent = `${script.title || 'Trailer outline'} — ${script.logline || ''}`;
    renderList(scriptVoiceover, script.voiceover || []);
    renderList(scriptTextCards, script.text_cards || []);
    renderList(scriptBeats, script.beats || []);
    const trace = result.agent_trace || [];
    agentSummary.textContent = `${result.iteration_count || 1} edit iteration(s) · ${result.analysis_provider || 'local'} · render QA ${result.render_report?.passed ? 'passed' : 'not available'}.`;
    renderList(agentTrace, trace.map((item) => `${item.stage}${item.iteration ? ` · iteration ${item.iteration}` : ''} — ${item.status}`));

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
musicInput.addEventListener('change', () => setSelectedMusic(musicInput.files[0]));

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
    const music = musicInput.files[0];
    const audioMode = audioModeInput.value;
    const youtube = youtubeUrl.value.trim();
    const theme = themeInput.value.trim();
    if (!file && !youtube) {
        showError('Please select a video file or paste a YouTube URL.');
        return;
    }
    if (file && youtube) {
        showError('Choose either a video file or a YouTube URL, not both.');
        return;
    }
    if (theme.length < 2) {
        showError('Please enter a theme or story idea.');
        themeInput.focus();
        return;
    }
    if (['music', 'mixed'].includes(audioMode) && !music) {
        showError('Upload a soundtrack for music-only or mixed audio mode.');
        musicInput.focus();
        return;
    }

    try {
        updateStages(0.01, youtube ? 'Downloading YouTube source' : 'Uploading source');
        let uploadData;
        if (youtube) {
            const urlFormData = new FormData();
            urlFormData.append('url', youtube);
            uploadData = await readJson(await fetch('/api/upload-url', { method: 'POST', body: urlFormData }));
        } else {
            const uploadFormData = new FormData();
            uploadFormData.append('file', file);
            uploadData = await readJson(await fetch('/api/upload', { method: 'POST', body: uploadFormData }));
        }

        updateStages(0.08, 'Source validated');
        let musicData = null;
        if (music && audioMode !== 'source') {
            updateStages(0.11, 'Uploading soundtrack');
            const musicFormData = new FormData();
            musicFormData.append('file', music);
            musicData = await readJson(await fetch('/api/upload-music', { method: 'POST', body: musicFormData }));
            const musicDuration = Number(musicData.metadata?.duration || 0);
            musicMeta.textContent = `${musicData.filename} · ${formatBytes(musicData.size)} · ${musicData.metadata?.codec || 'audio'} · ${musicDuration.toFixed(1)} seconds`;
        }
        const generationForm = new FormData();
        generationForm.append('video_id', uploadData.video_id);
        if (musicData) generationForm.append('music_id', musicData.music_id);
        generationForm.append('edit_prompt', theme);
        generationForm.append('audio_mode', audioMode);
        generationForm.append('target_duration', document.getElementById('targetDuration').value);
        generationForm.append('energy', document.getElementById('energy').value);
        generationForm.append('aspect_ratio', document.getElementById('aspectRatio').value);
        generationForm.append('strategy', document.getElementById('strategy').value);
        generationForm.append('analysis_mode', analysisModeInput.value);
        // For non-providers mode, the backend will ignore these and use defaults
        generationForm.append('transcription_provider', transcriptionProviderInput.value);
        generationForm.append('visual_provider', visualProviderInput.value);
        if (transcriptionProviderInput.value === 'elevenlabs' && elevenLabsKeyInput.value.trim()) {
            generationForm.append('elevenlabs_api_key', elevenLabsKeyInput.value.trim());
        }
        if (visualProviderInput.value === 'gemini' && geminiKeyInput.value.trim()) {
            generationForm.append('gemini_api_key', geminiKeyInput.value.trim());
        }
        const generationResponsePromise = fetch('/api/generate', { method: 'POST', body: generationForm });
        elevenLabsKeyInput.value = '';
        geminiKeyInput.value = '';
        const generationResponse = await generationResponsePromise;
        const generationData = await readJson(generationResponse);
        await pollJob(generationData.status_url);
    } catch (error) {
        elevenLabsKeyInput.value = '';
        geminiKeyInput.value = '';
        showError(error.message || 'The local cut failed.');
    }
});

transcriptionProviderInput.addEventListener('change', updateProviderFields);
visualProviderInput.addEventListener('change', updateProviderFields);
analysisModeInput.addEventListener('change', updateProviderFields);
updateProviderFields();
setStatus('Waiting for a video.', 0, 'STANDBY');
