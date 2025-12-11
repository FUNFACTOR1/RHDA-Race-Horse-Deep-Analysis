// --- Elementi DOM ---
const imageUploadInput = document.getElementById('image-upload-input');
const fileNameDisplay = document.getElementById('file-name-display');
const horseSelectionSection = document.getElementById('horse-selection-section');
const statusMessagePreprocessing = document.getElementById('status-message-preprocessing');
const processedHorsesContainer = document.getElementById('processed-horses-container');
const likelihoodThresholdSlider = document.getElementById('likelihood-threshold-slider');
const likelihoodThresholdValueDisplay = document.getElementById('likelihood-threshold-value');
const analysisResultsSection = document.getElementById('analysis-results-section');
const statusMessageAnalysis = document.getElementById('status-message-analysis');
const finalHorseImageContainer = document.getElementById('final-horse-image-container');
const anglesResultsList = document.getElementById('angles-results-list');
const distancesResultsList = document.getElementById('distances-results-list');
const keypointsJsonOutput = document.getElementById('keypoints-json-output');
const biomechanicsJsonOutput = document.getElementById('biomechanics-json-output');

const videoUploadInput = document.getElementById('video-upload-input');
const videoFileNameDisplay = document.getElementById('video-file-name-display');
const startVideoAnalysisButton = document.getElementById('start-video-analysis-button');
const videoProcessingSection = document.getElementById('video-processing-section');
const statusMessageVideo = document.getElementById('status-message-video');
const videoAnalysisResultsSection = document.getElementById('video-analysis-results-section');
const videoPlayer = document.getElementById('analyzed-video-player');
const videoKeypointsCanvas = document.getElementById('video-keypoints-canvas');
const videoKeypointsJsonOutput = document.getElementById('video-keypoints-json-output');
const videoTaskFinalStatusMessage = document.getElementById('video-task-final-status-message');

// NUOVI Elementi DOM per Biomeccanica Avanzata Video
const advancedVideoBiomechanicsContainer = document.getElementById('advanced-video-biomechanics-container');
const videoAdvTemporoSpazialiList = document.getElementById('video-adv-temporo-spaziali-list');
const videoAdvRomArticolariList = document.getElementById('video-adv-rom-articolari-list');
const videoAdvSimmetriaList = document.getElementById('video-adv-simmetria-list');
const videoAdvEfficienzaScheletrica = document.getElementById('video-adv-efficienza-scheletrica');
const videoAdvAvvisiList = document.getElementById('video-adv-avvisi-list');
const videoAdvancedBiomechanicsJsonOutput = document.getElementById('video-advanced-biomechanics-json-output');


// --- HUGGING FACE & URLs ---
const HF_ACCESS_TOKEN = 'hf_.....................................'; //

// URLs per i servizi su Hugging Face (MS1, MS2-Immagine, MS3 rimangono su HF per questo test)
const URL_MS1_PREPROCESSING = 'https://TURBOZAMPI-rhda-image-preprocessor.hf.space/process_image_for_selection';
const URL_MS2_DLC_KEYPOINTS_IMAGE = 'https://TURBOZAMPI-RHDAZAMPIMOD.hf.space/estimate_pose/';
const URL_MS3_BIOMECHANICS = 'https://TURBOZAMPI-rhda-biomechanics-analyzer.hf.space/analyze_biomechanics';

// --- MODIFICA PER TEST LOCALE DI RHDAV ---
// Commenta le URL originali di RHDAV su Hugging Face:
// const URL_MS2_VIDEO_ANALYZE = 'https://TURBOZAMPI-RHDAV.hf.space/analyze_video_segment/'; 
// const URL_MS2_VIDEO_STATUS_BASE = 'https://TURBOZAMPI-RHDAV.hf.space/task_status/'; 

// Usa queste URL per puntare al servizio RHDAV in esecuzione localmente su porta 7860:
const URL_MS2_VIDEO_ANALYZE = 'http://localhost:7860/analyze_video_segment/';
const URL_MS2_VIDEO_STATUS_BASE = 'http://localhost:7860/task_status/';
// --- FINE MODIFICA PER TEST LOCALE ---

// --- Parametri Default ---
const MS2_DLC_MODEL_NAME = "superanimal_quadruped_dlcrnet";
const MS3_MIN_CONFIDENCE_FOR_CALCULATION = 0.5;
const MS3_PIXELS_PER_METER = null;
const MS1_DEFAULT_PROCESSING_OPTIONS = {
    yolo_confidence_threshold: 0.25, max_horses_to_return: 5, apply_rembg_after_yolo_crop: true,
    rembg_model_name: "u2net_horse", apply_clahe: true, clahe_clip_limit: 2.0,
    clahe_tile_grid_size_rows: 8, clahe_tile_grid_size_cols: 8, apply_sharpening: true, sharpen_intensity: 1.2
};
const TARGET_EQUINE_KEYPOINT_NAMES = ["Nose", "Eye", "Nearknee", "Nearfrontfetlock", "Nearfrontfoot","Offknee", "Offfrontfetlock", "Offfrontfoot", "Shoulder", "Midshoulder","Elbow", "Girth", "Wither", "Nearhindhock", "Nearhindfetlock","Nearhindfoot", "Hip", "Stifle", "Offhindhock", "Offhindfetlock","Offhindfoot", "Ischium"];
const SKELETON_PAIRS = [["Nose", "Eye"], ["Eye", "Midshoulder"], ["Midshoulder", "Wither"], ["Wither", "Girth"],["Girth", "Hip"], ["Hip", "Ischium"], ["Wither", "Shoulder"], ["Shoulder", "Elbow"],["Elbow", "Nearknee"], ["Nearknee", "Nearfrontfetlock"], ["Nearfrontfetlock", "Nearfrontfoot"],["Wither", "Offknee"], ["Offknee", "Offfrontfetlock"], ["Offfrontfetlock", "Offfrontfoot"],["Hip", "Stifle"], ["Stifle", "Nearhindhock"], ["Nearhindhock", "Nearhindfetlock"],["Nearhindfetlock", "Nearhindfoot"], ["Hip", "Offhindhock"], ["Offhindhock", "Offhindfetlock"],["Offhindfetlock", "Offhindfoot"],];
const KEYPOINT_COLORS = {Nose: '#FF6347', Eye: '#FFD700', Nearknee: '#ADFF2F', Nearfrontfetlock: '#ADFF2F', Nearfrontfoot: '#ADFF2F',Offknee: '#32CD32', Offfrontfetlock: '#32CD32', Offfrontfoot: '#32CD32', Shoulder: '#FF4500', Midshoulder: '#FF8C00',Elbow: '#FF4500', Girth: '#DA70D6', Wither: '#DA70D6', Nearhindhock: '#00CED1', Nearhindfetlock: '#00CED1',Nearhindfoot: '#00CED1', Hip: '#BA55D3', Stifle: '#9370DB', Offhindhock: '#48D1CC', Offhindfetlock: '#48D1CC',Offhindfoot: '#48D1CC', Ischium: '#BA55D3'};
const DEFAULT_KEYPOINT_COLOR = '#FF00FF';
const SKELETON_COLOR = 'rgba(255, 255, 255, 0.8)';

let currentUploadedImageFile = null;
let processedHorseImagesData = [];
let selectedHorseData = null;
let currentRawKeypointsFromMS2 = null;
let currentUploadedFrames = [];
let currentVideoTaskId = null;
let videoKeypointsPerFrameData = null;
let videoPlayerNaturalWidth = null;
let videoPlayerNaturalHeight = null;
let pollingIntervalId = null;
const POLLING_INTERVAL_MS = 5000;

const apiHeaders = new Headers();
if (HF_ACCESS_TOKEN && HF_ACCESS_TOKEN.startsWith('hf_')) {
    apiHeaders.append('Authorization', `Bearer ${HF_ACCESS_TOKEN}`);
} else {
    console.warn("HF_ACCESS_TOKEN non configurato o non valido.");
}

function logger(...args) {
    console.log('[RHDA Frontend]', ...args);
}

function setStatus(element, message, type = 'info') {
    if (element) {
        element.textContent = message;
        element.className = `status status-${type}`; // Assicura che la classe 'status' sia sempre presente
        element.style.display = 'block';
    }
    logger(`Status (element: ${element ? element.id : 'N/A'}, type: ${type}): ${message}`);
}


async function base64ToFile(base64, filename, mimeType) {
    try {
        const res = await fetch(base64);
        const blob = await res.blob();
        return new File([blob], filename, { type: mimeType });
    } catch (error) {
        logger("Errore in base64ToFile:", error); throw error;
    }
}

function resetImageUIState(resetFileSelection = true) {
    if (resetFileSelection) {
        if (imageUploadInput) imageUploadInput.value = '';
        if (fileNameDisplay) fileNameDisplay.textContent = 'Nessun file immagine selezionato';
        currentUploadedImageFile = null;
    }
    if (horseSelectionSection) horseSelectionSection.style.display = 'none';
    if (analysisResultsSection) analysisResultsSection.style.display = 'none';
    if (processedHorsesContainer) processedHorsesContainer.innerHTML = '<p class="placeholder">Nessun cavallo rilevato.</p>';
    if (finalHorseImageContainer) finalHorseImageContainer.innerHTML = '<p class="placeholder">L\'immagine annotata apparirà qui.</p>';
    if (anglesResultsList) anglesResultsList.innerHTML = '<li>Nessun dato angolare.</li>';
    if (distancesResultsList) distancesResultsList.innerHTML = '<li>Nessun dato distanza.</li>';
    if (keypointsJsonOutput) keypointsJsonOutput.textContent = 'Nessun dato keypoint disponibile.';
    if (biomechanicsJsonOutput) biomechanicsJsonOutput.textContent = 'Nessun dato biomeccanico disponibile.';
    setStatus(statusMessagePreprocessing, 'Pronto per caricare un file immagine...', 'idle');
    setStatus(statusMessageAnalysis, 'In attesa di analisi immagine...', 'idle');
    selectedHorseData = null;
    processedHorseImagesData = [];
    currentRawKeypointsFromMS2 = null;
}

function resetVideoUIState(resetFileSelection = true) {
    if (resetFileSelection) {
        if (videoUploadInput) videoUploadInput.value = '';
        if (videoFileNameDisplay) videoFileNameDisplay.textContent = 'Nessuna sequenza di frame selezionata';
        if (startVideoAnalysisButton) startVideoAnalysisButton.style.display = 'none';
        currentUploadedFrames = [];
    }
    if (videoProcessingSection) videoProcessingSection.style.display = 'block'; // Mantieni visibile la card principale
    if (videoAnalysisResultsSection) videoAnalysisResultsSection.style.display = 'none'; // Nascondi i risultati specifici

    if (videoPlayer) {
        videoPlayer.removeEventListener('timeupdate', drawKeypointsOnVideoFrame);
        videoPlayer.removeEventListener('seeked', drawKeypointsOnVideoFrame);
        videoPlayer.removeEventListener('play', drawKeypointsOnVideoFrame);
        videoPlayer.removeEventListener('pause', handleVideoPauseForRedraw);
        videoPlayer.removeEventListener('loadeddata', handleVideoLoadedData);
        // videoPlayer.removeEventListener('error', handleVideoPlayerError); 
        videoPlayer.pause();
        videoPlayer.src = '';
        videoPlayer.load();
        if (videoPlayer.parentElement) videoPlayer.parentElement.style.display = 'none'; // Nascondi il wrapper del player
    }
    if (videoKeypointsCanvas) {
        const ctx = videoKeypointsCanvas.getContext('2d');
        ctx.clearRect(0, 0, videoKeypointsCanvas.width, videoKeypointsCanvas.height);
    }
    if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = 'Nessun dato keypoint video disponibile.';
    if (videoTaskFinalStatusMessage) {
        videoTaskFinalStatusMessage.innerHTML = '';
        videoTaskFinalStatusMessage.className = 'status';
        videoTaskFinalStatusMessage.style.display = 'none';
    }
    
    // Reset della nuova sezione Biomeccanica Avanzata Video
    if (advancedVideoBiomechanicsContainer) advancedVideoBiomechanicsContainer.style.display = 'none';
    if (videoAdvTemporoSpazialiList) videoAdvTemporoSpazialiList.innerHTML = '<li>Nessun dato calcolato.</li>';
    if (videoAdvRomArticolariList) videoAdvRomArticolariList.innerHTML = '<li>Nessun dato calcolato.</li>';
    if (videoAdvSimmetriaList) videoAdvSimmetriaList.innerHTML = '<li>Nessun dato calcolato.</li>';
    if (videoAdvEfficienzaScheletrica) videoAdvEfficienzaScheletrica.textContent = 'N/D';
    if (videoAdvAvvisiList) videoAdvAvvisiList.innerHTML = '<li>Nessun avviso.</li>';
    if (videoAdvancedBiomechanicsJsonOutput) videoAdvancedBiomechanicsJsonOutput.textContent = 'Nessun dato disponibile.';

    setStatus(statusMessageVideo, 'Pronto per caricare una sequenza di frame...', 'idle');
    if (pollingIntervalId) { clearTimeout(pollingIntervalId); pollingIntervalId = null; }
    currentVideoTaskId = null;
    videoKeypointsPerFrameData = null;
    videoPlayerNaturalWidth = null;
    videoPlayerNaturalHeight = null;
}

// --- Logica Analisi Immagine Singola ---
if (imageUploadInput) {
    imageUploadInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files.length > 0) {
            currentUploadedImageFile = files[0];
            if (fileNameDisplay) fileNameDisplay.textContent = `File Immagine: ${currentUploadedImageFile.name}`;
            resetImageUIState(false); // Non resettare la selezione file appena fatta
            resetVideoUIState(true);  // Resetta completamente la UI video
            if (videoProcessingSection) videoProcessingSection.style.display = 'none'; // Nascondi la card video se si carica immagine
            setStatus(statusMessagePreprocessing, `Pronto per pre-elaborare '${currentUploadedImageFile.name}'. Elaborazione automatica avviata...`, 'idle');
            handleImagePreprocessing();
        }
    });
}

async function handleImagePreprocessing() {
    if (!currentUploadedImageFile) {
        setStatus(statusMessagePreprocessing, 'Errore: Nessun file immagine selezionato.', 'error');
        return;
    }
    if (horseSelectionSection) horseSelectionSection.style.display = 'block';
    if (analysisResultsSection) analysisResultsSection.style.display = 'none';
    setStatus(statusMessagePreprocessing, `Pre-elaborazione di '${currentUploadedImageFile.name}' in corso...`, 'processing');
    if (processedHorsesContainer) processedHorsesContainer.innerHTML = '<div class="loader"></div><p class="placeholder" style="text-align:center;">Elaborazione...</p>';

    const formData = new FormData();
    formData.append('file', currentUploadedImageFile);
    Object.entries(MS1_DEFAULT_PROCESSING_OPTIONS).forEach(([k, v]) => {
        if (typeof v === 'boolean') formData.append(k, String(v).toLowerCase());
        else if (v !== null && v !== undefined) formData.append(k, v);
    });

    try {
        const response = await fetch(URL_MS1_PREPROCESSING, { method: 'POST', headers: apiHeaders, body: formData });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Errore MS1 (${response.status}): ${errorText.substring(0, 100)}`);
        }
        const result = await response.json();
        processedHorseImagesData = result.processed_horses || [];
        displaySelectableHorses(processedHorseImagesData);
        setStatus(statusMessagePreprocessing, processedHorseImagesData.length > 0 ? `Pre-elaborazione completata. Seleziona un cavallo.` : (result.message || 'Nessun cavallo rilevato.'), processedHorseImagesData.length > 0 ? 'success' : 'warning');
        if (processedHorseImagesData.length === 0 && processedHorsesContainer) processedHorsesContainer.innerHTML = '<p class="placeholder">Nessun cavallo rilevato.</p>';
    } catch (error) {
        console.error('Errore durante la pre-elaborazione (MS1):', error);
        setStatus(statusMessagePreprocessing, `Errore Pre-elaborazione: ${error.message}`, 'error');
        if (processedHorsesContainer) processedHorsesContainer.innerHTML = '<p class="placeholder">Errore nel rilevamento.</p>';
    }
}

function displaySelectableHorses(horsesData) {
    if (!processedHorsesContainer) return;
    processedHorsesContainer.innerHTML = '';
    if (!horsesData || horsesData.length === 0) {
        processedHorsesContainer.innerHTML = '<p class="placeholder">Nessun cavallo da selezionare.</p>';
        return;
    }
    horsesData.forEach(horse => {
        const div = document.createElement('div');
        div.classList.add('horse-selectable-item');
        div.setAttribute('data-horse-id', horse.horse_id);
        const img = document.createElement('img');
        img.src = horse.cropped_image_base64;
        img.alt = `Cavallo Rilevato ${horse.horse_id}`;
        const p = document.createElement('p');
        p.textContent = `Cavallo ID: ${horse.horse_id}`;
        div.appendChild(img);
        div.appendChild(p);
        div.addEventListener('click', () => handleHorseSelectionForAnalysis(horse));
        processedHorsesContainer.appendChild(div);
    });
}

async function handleHorseSelectionForAnalysis(horseDataObject) {
    selectedHorseData = horseDataObject;
    logger(`Cavallo selezionato per analisi: ID ${selectedHorseData.horse_id}`);
    if (analysisResultsSection) analysisResultsSection.style.display = 'block';
    if (finalHorseImageContainer) finalHorseImageContainer.innerHTML = '<div class="loader"></div><p class="placeholder" style="text-align:center;">Analisi immagine in corso...</p>';
    if (anglesResultsList) anglesResultsList.innerHTML = '<li>In attesa...</li>';
    if (distancesResultsList) distancesResultsList.innerHTML = '<li>In attesa...</li>';
    if (keypointsJsonOutput) keypointsJsonOutput.textContent = 'In attesa dei dati da MS2...';
    if (biomechanicsJsonOutput) biomechanicsJsonOutput.textContent = 'In attesa dei dati da MS3...';
    await performMS2ImageAnalysis();
}

async function performMS2ImageAnalysis() {
    if (!selectedHorseData || !selectedHorseData.cropped_image_base64) {
        setStatus(statusMessageAnalysis, 'Dati cavallo mancanti.', 'error'); return;
    }
    setStatus(statusMessageAnalysis, `Analisi MS2 per Cavallo ID: ${selectedHorseData.horse_id} in corso...`, 'processing');
    try {
        const imageFileForDLC = await base64ToFile(selectedHorseData.cropped_image_base64, `${selectedHorseData.horse_id}_from_ms1.png`, 'image/png');
        const dlcFormData = new FormData();
        dlcFormData.append('image_file', imageFileForDLC);
        dlcFormData.append('model_name', MS2_DLC_MODEL_NAME);
        const currentLikelihoodThreshold = likelihoodThresholdSlider ? parseFloat(likelihoodThresholdSlider.value) : 0.20;
        dlcFormData.append('likelihood_threshold', String(currentLikelihoodThreshold));

        const dlcResponse = await fetch(URL_MS2_DLC_KEYPOINTS_IMAGE, { method: 'POST', headers: apiHeaders, body: dlcFormData });
        const dlcResult = await dlcResponse.json();

        if (!dlcResponse.ok) {
            throw new Error(`Errore da MS2 (${dlcResponse.status}): ${JSON.stringify(dlcResult.detail || dlcResult)}`);
        }
        logger("Risultato da MS2 (immagine):", dlcResult);
        currentRawKeypointsFromMS2 = dlcResult.keypoints;

        if (finalHorseImageContainer) {
            finalHorseImageContainer.innerHTML = '';
            if (dlcResult.annotated_image_base64) {
                const imgDisplay = document.createElement('img');
                imgDisplay.id = 'analysisDisplayImage';
                imgDisplay.alt = `Cavallo Analizzato ${selectedHorseData.horse_id}`;
                imgDisplay.src = dlcResult.annotated_image_base64;
                imgDisplay.style.maxWidth = '100%';
                imgDisplay.style.maxHeight = '100%';
                imgDisplay.style.objectFit = 'contain';
                finalHorseImageContainer.appendChild(imgDisplay);
            } else {
                finalHorseImageContainer.innerHTML = '<p class="placeholder error">Immagine annotata non ricevuta da MS2.</p>';
            }
        }
        if (keypointsJsonOutput) keypointsJsonOutput.textContent = JSON.stringify(currentRawKeypointsFromMS2 || {error: "Nessun keypoint da MS2"}, null, 2);

        if (!Array.isArray(currentRawKeypointsFromMS2) || currentRawKeypointsFromMS2.length === 0) {
            setStatus(statusMessageAnalysis, `MS2: Nessun keypoint ricevuto. Impossibile procedere con MS3.`, 'warning');
            displayBiomechanicsResults(null); // Per l'immagine singola
            return;
        }
        setStatus(statusMessageAnalysis, `Keypoint da MS2: ${currentRawKeypointsFromMS2.length}. Avvio MS3...`, 'processing');
        await performMS3Analysis(currentRawKeypointsFromMS2, selectedHorseData.horse_id);
    } catch (error) {
        console.error('Errore analisi immagine (MS2):', error);
        setStatus(statusMessageAnalysis, `Errore MS2: ${error.message}`, 'error');
        if (finalHorseImageContainer) finalHorseImageContainer.innerHTML = `<p class="placeholder error">Errore MS2.</p>`;
        displayBiomechanicsResults(null); // Per l'immagine singola
    }
}

async function performMS3Analysis(keypointsForBiomechanics, imageId) { // Analisi per immagine singola
    const keypointsPayload = keypointsForBiomechanics.map(kp => ({
        label: kp.keypoint || kp.name,
        x: kp.x, y: kp.y,
        confidence: kp.score !== undefined ? kp.score : (kp.likelihood !== undefined ? kp.likelihood : 0)
    }));

    const payload = {
        image_identifier: imageId,
        keypoints_data: { keypoints: keypointsPayload },
        min_confidence_for_calculation: MS3_MIN_CONFIDENCE_FOR_CALCULATION,
        pixels_per_meter: MS3_PIXELS_PER_METER
    };
    if (biomechanicsJsonOutput) biomechanicsJsonOutput.textContent = 'Analisi biomeccanica (MS3) in corso...';
    try {
        const headersMS3 = new Headers(apiHeaders);
        headersMS3.append('Content-Type', 'application/json');
        const response = await fetch(URL_MS3_BIOMECHANICS, { method: 'POST', headers: headersMS3, body: JSON.stringify(payload) });
        const result = await response.json();
        logger("Risultato JSON GREZZO da MS3 (biomeccanica immagine):", JSON.stringify(result, null, 2));

        if (!response.ok) {
            throw new Error(`Errore da MS3 (${response.status}): ${JSON.stringify(result.detail || result)}`);
        }
        
        if (biomechanicsJsonOutput) biomechanicsJsonOutput.textContent = JSON.stringify(result, null, 2);
        displayBiomechanicsResults(result); // Per l'immagine singola
        
        setStatus(statusMessageAnalysis, `Analisi biomeccanica immagine completata per ID: ${imageId}.`, 'success');
    } catch (error) {
        console.error('Errore analisi biomeccanica immagine (MS3):', error);
        setStatus(statusMessageAnalysis, `Errore MS3 Immagine: ${error.message}`, 'error');
        if (biomechanicsJsonOutput) biomechanicsJsonOutput.textContent = `Errore MS3 Immagine: ${error.message}`;
        displayBiomechanicsResults(null); // Per l'immagine singola
    }
}

function displayBiomechanicsResults(ms3Result) { // Per l'immagine singola
    logger("displayBiomechanicsResults (immagine singola) chiamata con:", ms3Result);
    if (anglesResultsList) anglesResultsList.innerHTML = '';
    if (distancesResultsList) distancesResultsList.innerHTML = '';

    let foundAngles = false;
    if (ms3Result && ms3Result.angles && Array.isArray(ms3Result.angles)) {
        ms3Result.angles.forEach((angleItem) => {
            const name = angleItem.name;
            const value = angleItem.value_degrees;
            if (typeof name === 'string' && name.trim() !== "" && typeof value === 'number' && !isNaN(value)) {
                const li = document.createElement('li');
                li.innerHTML = `<span class="metric-name">${name}:</span> <span class="metric-value">${value.toFixed(1)}°</span>`;
                if (anglesResultsList) anglesResultsList.appendChild(li);
                foundAngles = true;
            }
        });
    }
    if (!foundAngles && anglesResultsList) {
        anglesResultsList.innerHTML = '<li>Nessun dato angolare calcolato o valido.</li>';
    }

    let foundDistances = false;
    if (ms3Result && ms3Result.distances && Array.isArray(ms3Result.distances)) {
        ms3Result.distances.forEach((distItem) => {
            const name = distItem.name;
            const value = distItem.value_pixels; // o value_meters se disponibile e preferito
            if (typeof name === 'string' && name.trim() !== "" && typeof value === 'number' && !isNaN(value)) {
                const li = document.createElement('li');
                // Potresti aggiungere logica per mostrare metri se MS3_PIXELS_PER_METER è stato usato
                li.innerHTML = `<span class="metric-name">${name}:</span> <span class="metric-value">${value.toFixed(2)} unità pixel</span>`;
                if (distancesResultsList) distancesResultsList.appendChild(li);
                foundDistances = true;
            }
        });
    }
    if (!foundDistances && distancesResultsList) {
        distancesResultsList.innerHTML = '<li>Nessun dato distanza calcolato o valido.</li>';
    }
}


if (likelihoodThresholdSlider && likelihoodThresholdValueDisplay) {
    likelihoodThresholdSlider.addEventListener('input', async (event) => {
        const newValue = parseFloat(event.target.value).toFixed(2);
        likelihoodThresholdValueDisplay.textContent = newValue;
        // Se un'immagine singola è stata analizzata, riesegui MS2 e MS3
        if (selectedHorseData && selectedHorseData.cropped_image_base64 && currentRawKeypointsFromMS2) {
            logger("Soglia cambiata, rieseguo analisi MS2 e MS3 per immagine singola.");
            await performMS2ImageAnalysis(); 
        }
        // Se i dati dei keypoint video sono caricati, ridisegna i keypoint sul video
        if (videoKeypointsPerFrameData && videoPlayer && videoKeypointsCanvas) {
            logger("Soglia cambiata, ridisegno keypoints su video.");
            drawKeypointsOnVideoFrame();
        }
    });
}

// --- Logica Video (Sequenza di Frame) ---
if (videoUploadInput) {
    videoUploadInput.addEventListener('change', (event) => {
        const files = event.target.files;
        logger("File selezionati per la sequenza video:", files);
        if (files && files.length > 0) {
            currentUploadedFrames = Array.from(files).sort((a, b) =>
                a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })
            );
            logger("currentUploadedFrames dopo selezione e ordinamento:", currentUploadedFrames.map(f => f.name));
            if (videoFileNameDisplay) {
                videoFileNameDisplay.textContent = currentUploadedFrames.length === 1 ?
                    `1 frame selezionato: ${currentUploadedFrames[0].name}` :
                    `${currentUploadedFrames.length} frame selezionati (da ${currentUploadedFrames[0].name} a ${currentUploadedFrames[currentUploadedFrames.length - 1].name})`;
            }
            if (startVideoAnalysisButton) startVideoAnalysisButton.style.display = 'inline-block';
            resetVideoUIState(false); // Non resettare la selezione file
            resetImageUIState(true);  // Resetta la UI immagine
            if (videoProcessingSection) videoProcessingSection.style.display = 'block'; // Assicura che la card video sia visibile
            if (horseSelectionSection) horseSelectionSection.style.display = 'none'; // Nascondi selezione cavallo immagine
            if (analysisResultsSection) analysisResultsSection.style.display = 'none'; // Nascondi risultati immagine

            setStatus(statusMessageVideo, `Pronto per analizzare ${currentUploadedFrames.length} frame. Clicca 'Avvia Analisi Sequenza'.`, 'idle');
        } else {
            logger("Nessun file selezionato o selezione cancellata.");
            currentUploadedFrames = [];
            if (videoFileNameDisplay) videoFileNameDisplay.textContent = 'Nessuna sequenza di frame selezionata';
            if (startVideoAnalysisButton) startVideoAnalysisButton.style.display = 'none';
        }
    });
}

if (startVideoAnalysisButton) {
    startVideoAnalysisButton.addEventListener('click', async () => {
        logger("Bottone 'Avvia Analisi Sequenza' cliccato.");
        if (currentUploadedFrames.length === 0) {
            setStatus(statusMessageVideo, 'Errore: Nessuna sequenza di frame selezionata per l\'analisi.', 'error');
            logger("Tentativo di avviare analisi video senza frame selezionati.");
            return;
        }
        await handleVideoAnalysis();
    });
}

async function handleVideoAnalysis() {
    logger("handleVideoAnalysis avviata.");
    if (currentUploadedFrames.length === 0) {
        logger("handleVideoAnalysis: currentUploadedFrames è vuoto. Uscita anticipata.");
        setStatus(statusMessageVideo, 'Errore interno: nessun frame da inviare.', 'error');
        return;
    }

    // Reset specifico prima di un nuovo avvio analisi video
    if (videoAnalysisResultsSection) videoAnalysisResultsSection.style.display = 'none';
    if (advancedVideoBiomechanicsContainer) advancedVideoBiomechanicsContainer.style.display = 'none';
    if (videoTaskFinalStatusMessage) {
        videoTaskFinalStatusMessage.innerHTML = '';
        videoTaskFinalStatusMessage.style.display = 'none';
    }
    if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = 'In attesa dei dati dal server...';
    if (videoPlayer && videoPlayer.parentElement) videoPlayer.parentElement.style.display = 'none';
    
    setStatus(statusMessageVideo, `Upload e analisi di ${currentUploadedFrames.length} frame in corso...`, 'processing');
    logger(`Numero di frame da processare: ${currentUploadedFrames.length}`);

    const formData = new FormData();
    currentUploadedFrames.forEach((file) => {
        formData.append('frame_files', file, file.name);
    });

    if (Array.from(formData.getAll('frame_files')).length === 0) {
        logger("ERRORE CRITICO: Nessun file in FormData da inviare per l'analisi video.");
        setStatus(statusMessageVideo, `Errore interno: nessun frame valido da inviare.`, 'error');
        return;
    }

    try {
        logger(`Invio richiesta a ${URL_MS2_VIDEO_ANALYZE}`);
        const response = await fetch(URL_MS2_VIDEO_ANALYZE, { method: 'POST', headers: apiHeaders, body: formData });
        logger(`Risposta ricevuta da ${URL_MS2_VIDEO_ANALYZE}, status: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            let errorDetail = `Errore HTTP ${response.status} (${response.statusText})`;
            try {
                const errorJson = JSON.parse(errorText);
                if (errorJson.detail && Array.isArray(errorJson.detail)) {
                    errorDetail = `Errore di validazione dal server: ${errorJson.detail.map(e => `campo '${e.loc.join('.')}' - ${e.msg}`).join('; ')}`;
                } else if (errorJson.detail) {
                    errorDetail = errorJson.detail;
                } else {
                    errorDetail = errorText.substring(0, 300); 
                }
            } catch (e) {
                errorDetail = errorText.substring(0, 300) || `Errore HTTP ${response.status}`;
            }
            logger(`Errore fetch: ${errorDetail}`);
            throw new Error(errorDetail);
        }
        const result = await response.json();
        currentVideoTaskId = result.task_id;
        logger(`Analisi sequenza frame avviata con successo. Task ID: ${currentVideoTaskId}`);
        setStatus(statusMessageVideo, `Task ${currentVideoTaskId} accodato. Controllo stato...`, 'processing');
        pollVideoTaskStatus(currentVideoTaskId);
    } catch (error) {
        console.error('Errore durante handleVideoAnalysis (fetch o gestione risposta):', error);
        setStatus(statusMessageVideo, `Errore avvio analisi: ${error.message}`, 'error');
        if (videoTaskFinalStatusMessage) {
            setStatus(videoTaskFinalStatusMessage, `Errore avvio analisi: ${error.message}`, 'error');
        }
    }
}

function pollVideoTaskStatus(taskId) {
    if (!taskId) { logger("pollVideoTaskStatus: taskId nullo."); return; }
    if (pollingIntervalId) clearTimeout(pollingIntervalId); 

    logger(`Polling per Task ID: ${taskId} a ${URL_MS2_VIDEO_STATUS_BASE}${taskId}`);
    fetch(`${URL_MS2_VIDEO_STATUS_BASE}${taskId}`, { headers: apiHeaders, cache: 'no-store' })
        .then(response => {
            logger(`Risposta polling per ${taskId}: Status ${response.status}, OK: ${response.ok}`);
            if (!response.ok) {
                if (response.status === 404) {
                    setStatus(statusMessageVideo, `Task ${taskId} non trovato. Polling interrotto.`, 'error');
                    if (videoTaskFinalStatusMessage) setStatus(videoTaskFinalStatusMessage, `Task ${taskId} non trovato.`, 'error');
                    return null; 
                }
                throw new Error(`Errore HTTP ${response.status} durante polling.`);
            }
            return response.json();
        })
        .then(async data => { // Aggiunto async qui per await interno
            if (!data) return; 

            logger(`Dati stato task ${taskId}:`, data);
            let progressPercent = data.progress !== undefined ? parseFloat(data.progress).toFixed(0) : "N/D";
            let statusText = `Stato Task ${taskId}: ${data.status || 'N/D'}. Progresso: ${progressPercent}%`;
            
            let statusType = 'processing';
            if (data.status === 'failed') statusType = 'error';
            else if (data.status === 'completed') statusType = 'success';
            setStatus(statusMessageVideo, statusText, statusType);

            if (data.status === 'completed' || data.status === 'failed') {
                if (videoAnalysisResultsSection) videoAnalysisResultsSection.style.display = 'block';
            }
            if (videoTaskFinalStatusMessage && (data.status === 'processing' || data.status === 'completed')) {
                videoTaskFinalStatusMessage.innerHTML = ''; 
                videoTaskFinalStatusMessage.style.display = 'none';
            }

            if (data.status === 'completed') {
                logger(`TASK ${taskId} COMPLETATO!`);
                logger(`  URL Video: ${data.result_video_url}`);
                logger(`  URL Dati JSON Keypoints: ${data.result_data_url}`);
                logger(`  URL Dati JSON Biomeccanica Avanzata: ${data.result_biomechanics_url}`); // NUOVO LOG

                if (data.result_video_url && videoPlayer) {
                    if (videoPlayer.parentElement) videoPlayer.parentElement.style.display = 'block';
                    videoPlayer.src = data.result_video_url; 
                    videoPlayer.load();
                } else {
                    logger("URL video non trovato o player non disponibile.");
                    if (videoTaskFinalStatusMessage) setStatus(videoTaskFinalStatusMessage, "Risultato video non disponibile.", 'warning');
                }

                if (data.result_data_url) {
                    try {
                        const resKeypoints = await fetch(data.result_data_url, {cache: 'no-store'});
                        logger(`Risposta Fetch per JSON Keypoints (${data.result_data_url}): Status ${resKeypoints.status}, OK: ${resKeypoints.ok}`);
                        if (!resKeypoints.ok) throw new Error(`Errore HTTP ${resKeypoints.status} caricamento dati keypoint.`);
                        const keypointData = await resKeypoints.json();
                        
                        videoKeypointsPerFrameData = keypointData;
                        logger("Dati keypoint video caricati:", videoKeypointsPerFrameData);
                        if(!videoKeypointsPerFrameData || !videoKeypointsPerFrameData.fps || !videoKeypointsPerFrameData.frames){
                            logger.error("Formato dati keypoint video non valido.", videoKeypointsPerFrameData);
                            if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = "Formato dati keypoint non valido.";
                            // Non ritornare, prova a caricare la biomeccanica
                        } else {
                           if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = "Dati keypoint caricati.";
                        }
                        
                        if (videoPlayer) {
                            ['loadeddata', 'timeupdate', 'seeked', 'play', 'pause'].forEach(event => 
                                videoPlayer.removeEventListener(event, event === 'loadeddata' ? handleVideoLoadedData : (event === 'pause' ? handleVideoPauseForRedraw : drawKeypointsOnVideoFrame))
                            );
                            videoPlayer.addEventListener('loadeddata', handleVideoLoadedData, { once: true });
                            videoPlayer.addEventListener('timeupdate', drawKeypointsOnVideoFrame);
                            videoPlayer.addEventListener('seeked', drawKeypointsOnVideoFrame);
                            videoPlayer.addEventListener('play', drawKeypointsOnVideoFrame);
                            videoPlayer.addEventListener('pause', handleVideoPauseForRedraw);
                            if (videoPlayer.readyState >= 2) handleVideoLoadedData(); // Se già caricato
                        }
                    } catch (err) {
                        logger("ERRORE caricamento/parsing dati keypoint JSON video:", err, `URL: ${data.result_data_url}`);
                        if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = `Errore caricamento dati keypoint: ${err.message}`;
                        if (videoTaskFinalStatusMessage) setStatus(videoTaskFinalStatusMessage, `Errore caricamento dati keypoint: ${err.message}`, 'error');
                    }
                } else {
                    logger("URL dati keypoint video non trovato.");
                    if (videoKeypointsJsonOutput) videoKeypointsJsonOutput.textContent = "Nessun URL per i dati keypoint.";
                }

                // GESTIONE NUOVA BIOMECCANICA AVANZATA VIDEO
                if (data.result_biomechanics_url) {
                    try {
                        const resBiomechanics = await fetch(data.result_biomechanics_url, {cache: 'no-store'});
                        logger(`Risposta Fetch per JSON Biomeccanica Avanzata (${data.result_biomechanics_url}): Status ${resBiomechanics.status}, OK: ${resBiomechanics.ok}`);
                        if (!resBiomechanics.ok) throw new Error(`Errore HTTP ${resBiomechanics.status} caricamento dati biomeccanica avanzata.`);
                        const advBiomechanicsData = await resBiomechanics.json();
                        logger("Dati biomeccanica avanzata video caricati:", advBiomechanicsData);
                        displayAdvancedVideoBiomechanicsResults(advBiomechanicsData);
                    } catch (err) {
                        logger("ERRORE caricamento/parsing dati biomeccanica avanzata JSON video:", err, `URL: ${data.result_biomechanics_url}`);
                        if (videoAdvancedBiomechanicsJsonOutput) videoAdvancedBiomechanicsJsonOutput.textContent = `Errore caricamento dati biomeccanica avanzata: ${err.message}`;
                        if (videoTaskFinalStatusMessage) setStatus(videoTaskFinalStatusMessage, `Errore caricamento dati biomeccanica avanzata: ${err.message}`, 'error');
                        displayAdvancedVideoBiomechanicsResults(null); // Mostra comunque la sezione con "Nessun dato"
                    }
                } else {
                    logger("URL dati biomeccanica avanzata video non trovato.");
                    displayAdvancedVideoBiomechanicsResults(null); // Mostra comunque la sezione con "Nessun dato"
                }

            } else if (data.status === 'failed') {
                if (videoTaskFinalStatusMessage) {
                    setStatus(videoTaskFinalStatusMessage, `Analisi fallita (Task ${taskId}): ${data.error_message || 'Dettagli non disponibili.'}`, 'error');
                }
                displayAdvancedVideoBiomechanicsResults(null); // Pulisci o mostra "Nessun dato"
            } else if (data.status === 'queued' || data.status === 'processing') {
                pollingIntervalId = setTimeout(() => pollVideoTaskStatus(taskId), POLLING_INTERVAL_MS);
            }
        })
        .catch(error => {
            logger('Errore grave durante polling task:', error);
            setStatus(statusMessageVideo, `Errore polling task ${taskId}: ${error.message}. Riprovo...`, 'error');
            // Considera una strategia di backoff più robusta se necessario
            pollingIntervalId = setTimeout(() => pollVideoTaskStatus(taskId), POLLING_INTERVAL_MS * 2); 
        });
}


// NUOVA FUNZIONE per visualizzare i risultati della biomeccanica avanzata del video
function displayAdvancedVideoBiomechanicsResults(data) {
    if (!advancedVideoBiomechanicsContainer) return;
    advancedVideoBiomechanicsContainer.style.display = 'block';

    if (videoAdvancedBiomechanicsJsonOutput) {
        videoAdvancedBiomechanicsJsonOutput.textContent = data ? JSON.stringify(data, null, 2) : 'Nessun dato JSON disponibile.';
    }

    // Funzione helper per popolare una lista
    const populateList = (listElement, items, categoryName, unit = '') => {
        if (!listElement) return;
        listElement.innerHTML = ''; // Pulisci la lista
        if (items && Array.isArray(items) && items.length > 0) {
            items.forEach(item => {
                const li = document.createElement('li');
                let valueDisplay = typeof item.value === 'number' ? item.value.toFixed(2) : item.value;
                li.innerHTML = `<span class="metric-name">${item.name || 'Sconosciuto'}:</span> <span class="metric-value">${valueDisplay} ${item.unit || unit}</span>`;
                listElement.appendChild(li);
            });
        } else {
            listElement.innerHTML = `<li>Nessun dato calcolato per ${categoryName}.</li>`;
        }
    };
    
    // Popola le varie sezioni
    populateList(videoAdvTemporoSpazialiList, data?.parametri_temporo_spaziali, 'Parametri Temporo-Spaziali');
    populateList(videoAdvRomArticolariList, data?.range_of_motion_articolari, 'Range of Motion Articolari', '°');
    populateList(videoAdvSimmetriaList, data?.indici_di_simmetria, 'Indici di Simmetria', '%');

    if (videoAdvEfficienzaScheletrica) {
        if (data && data.indice_efficienza_biomeccanica_scheletrica) {
            let efficienza = data.indice_efficienza_biomeccanica_scheletrica;
            videoAdvEfficienzaScheletrica.textContent = (typeof efficienza === 'object' && efficienza.value) ? efficienza.value : (typeof efficienza === 'string' || typeof efficienza === 'number' ? efficienza : 'N/D');
        } else {
            videoAdvEfficienzaScheletrica.textContent = 'N/D';
        }
    }

    if (videoAdvAvvisiList) {
        videoAdvAvvisiList.innerHTML = ''; // Pulisci
        if (data && data.avvisi_analisi_video && Array.isArray(data.avvisi_analisi_video) && data.avvisi_analisi_video.length > 0) {
            data.avvisi_analisi_video.forEach(avviso => {
                const li = document.createElement('li');
                li.textContent = avviso;
                videoAdvAvvisiList.appendChild(li);
            });
        } else if (data && typeof data.avvisi_analisi_video === 'string' && data.avvisi_analisi_video.trim() !== '') {
            const li = document.createElement('li');
            li.textContent = data.avvisi_analisi_video;
            videoAdvAvvisiList.appendChild(li);
        }
        else {
            videoAdvAvvisiList.innerHTML = '<li>Nessun avviso.</li>';
        }
    }
}


function handleVideoLoadedData() {
    logger("Video 'loadeddata' event.");
    if (!videoPlayer) return;
    videoPlayerNaturalWidth = videoPlayer.videoWidth;
    videoPlayerNaturalHeight = videoPlayer.videoHeight;
    if (videoPlayerNaturalWidth === 0 || videoPlayerNaturalHeight === 0) {
        logger("Attenzione: Dimensioni video naturali sono 0."); 
        if(videoTaskFinalStatusMessage) setStatus(videoTaskFinalStatusMessage, "Errore nel caricamento del video: dimensioni non valide.", "error");
        return;
    }
    logger(`Dimensioni video caricate: ${videoPlayerNaturalWidth}x${videoPlayerNaturalHeight}`);
    if (videoKeypointsCanvas) {
        videoKeypointsCanvas.width = videoPlayer.clientWidth; // Usa clientWidth per dimensioni reali nel DOM
        videoKeypointsCanvas.height = videoPlayer.clientHeight;
        logger(`Canvas ridimensionato a: ${videoKeypointsCanvas.width}x${videoKeypointsCanvas.height}`);
        drawKeypointsOnVideoFrame(); 
    }
}

function handleVideoPauseForRedraw() {
    logger("Video in pausa, ridisegno keypoints.");
    if(videoPlayer && videoPlayer.readyState >= 2) { // Assicurati che il video sia pronto
        drawKeypointsOnVideoFrame();
    }
}

function drawKeypointsOnVideoFrame() {
    if (!videoPlayer || !videoKeypointsCanvas || !videoKeypointsPerFrameData || !videoKeypointsPerFrameData.frames) return;
    if (!videoPlayerNaturalWidth || !videoPlayerNaturalHeight || videoPlayerNaturalWidth === 0 || videoPlayerNaturalHeight === 0) {
        // Prova a recuperare le dimensioni se non ancora impostate e il video è pronto
        if (videoPlayer.videoWidth > 0 && videoPlayer.videoHeight > 0) {
            videoPlayerNaturalWidth = videoPlayer.videoWidth;
            videoPlayerNaturalHeight = videoPlayer.videoHeight;
            logger(`Dimensioni video naturali recuperate in drawKeypoints: ${videoPlayerNaturalWidth}x${videoPlayerNaturalHeight}`);
        } else {
            logger("Dimensioni video naturali ancora non disponibili in drawKeypoints.");
            return;
        }
    }

    const ctx = videoKeypointsCanvas.getContext('2d');
    // Sincronizza le dimensioni del canvas con quelle del video player nel DOM
    if (videoKeypointsCanvas.width !== videoPlayer.clientWidth || videoKeypointsCanvas.height !== videoPlayer.clientHeight) {
        videoKeypointsCanvas.width = videoPlayer.clientWidth;
        videoKeypointsCanvas.height = videoPlayer.clientHeight;
        logger(`Canvas ridimensionato dinamicamente in drawKeypoints: ${videoKeypointsCanvas.width}x${videoKeypointsCanvas.height}`);
    }

    const scaleX = videoKeypointsCanvas.width / videoPlayerNaturalWidth;
    const scaleY = videoKeypointsCanvas.height / videoPlayerNaturalHeight;
    const threshold = likelihoodThresholdSlider ? parseFloat(likelihoodThresholdSlider.value) : 0.20;

    ctx.clearRect(0, 0, videoKeypointsCanvas.width, videoKeypointsCanvas.height);

    const fps = videoKeypointsPerFrameData.fps || 30; 
    const currentFrameIndex = Math.floor(videoPlayer.currentTime * fps);
    const frameKeypointsData = videoKeypointsPerFrameData.frames[currentFrameIndex.toString()];

    if (videoKeypointsJsonOutput) { // Mostra solo i keypoint del frame corrente
        videoKeypointsJsonOutput.textContent = JSON.stringify(frameKeypointsData || { note: `Frame ${currentFrameIndex} @ ${videoPlayer.currentTime.toFixed(2)}s (FPS: ${fps}) - Nessun keypoint per questa soglia o frame` }, null, 2);
    }

    if (!frameKeypointsData || !Array.isArray(frameKeypointsData)) return; 
    
    const keypointsMap = {};
    frameKeypointsData.forEach(kp => {
        if (kp && kp.keypoint && kp.score >= threshold) {
            keypointsMap[kp.keypoint] = { x: kp.x * scaleX, y: kp.y * scaleY, score: kp.score, name: kp.keypoint };
        }
    });

    ctx.strokeStyle = SKELETON_COLOR;
    ctx.lineWidth = Math.max(1, 2 * Math.min(scaleX, scaleY)); 
    SKELETON_PAIRS.forEach(pair => {
        const kp1 = keypointsMap[pair[0]];
        const kp2 = keypointsMap[pair[1]];
        if (kp1 && kp2) { ctx.beginPath(); ctx.moveTo(kp1.x, kp1.y); ctx.lineTo(kp2.x, kp2.y); ctx.stroke(); }
    });

    Object.values(keypointsMap).forEach(kp => {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, Math.max(2, 4 * Math.min(scaleX, scaleY)), 0, 2 * Math.PI); 
        ctx.fillStyle = KEYPOINT_COLORS[kp.name] || DEFAULT_KEYPOINT_COLOR;
        ctx.fill();
    });
}

// --- Inizializzazione ---
document.addEventListener('DOMContentLoaded', () => {
    logger("Frontend RHDA inizializzato.");
    if (likelihoodThresholdSlider && likelihoodThresholdValueDisplay) {
        likelihoodThresholdSlider.value = 0.20; // Valore di default
        likelihoodThresholdValueDisplay.textContent = parseFloat(likelihoodThresholdSlider.value).toFixed(2);
    }
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) uploadSection.style.display = 'block';
    
    resetImageUIState(); // Resetta la UI per l'analisi immagine
    resetVideoUIState(); // Resetta la UI per l'analisi video
    if (videoProcessingSection) videoProcessingSection.style.display = 'none'; // Nascondi la card video all'inizio
});
