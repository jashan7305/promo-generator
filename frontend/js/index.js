const uploadForm = document.getElementById('uploadForm');
const statusDiv = document.getElementById('status');
const downloadSection = document.getElementById('downloadSection');
const promoVideo = document.getElementById('promoVideo');
const downloadLink = document.getElementById('downloadLink');

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    statusDiv.textContent = "Uploading video...";
    downloadSection.style.display = "none";

    const fileInput = document.getElementById('videoFile');
    const themeInput = document.getElementById('theme');

    if (!fileInput.files.length) {
        alert("Please select a video file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Step 1: Upload the video
    const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    // console.log(uploadResponse);

    if (!uploadResponse.ok) {
        statusDiv.textContent = "Upload failed.";
        return;
    }

    const uploadData = await uploadResponse.json();
    const videoPath = uploadData.video_path;

    statusDiv.textContent = "Generating promo... (this may take a while)";

    // Step 2: Generate promo
    const generateForm = new FormData();
    generateForm.append('video_path', videoPath);
    generateForm.append('theme', themeInput.value);

    const generateResponse = await fetch('/api/generate', {
        method: 'POST',
        body: generateForm
    });
    // console.log(generateResponse)

    if (!generateResponse.ok) {
        const error = await generateResponse.json();
        statusDiv.textContent = "Error: " + error.detail;
        return;
    }

    const generateData = await generateResponse.json();
    const downloadUrl = generateData.download_url;

    statusDiv.textContent = "Promo generated successfully!";
    downloadSection.style.display = "block";

    promoVideo.src = downloadUrl;
    downloadLink.href = downloadUrl;
    downloadLink.textContent = "Download Promo";
});