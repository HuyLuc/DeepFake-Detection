/**
 * DeepFake Detection Web App V2.0 - Main JavaScript
 * API integration, file upload handling, result rendering
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const API_BASE = '/api';
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB

// Global state
let currentFile = null;
let batchFiles = []; // For batch processing
let isBatchProcessing = false;
let currentHistoryId = null;
let historyPage = 1;
let historyFilters = {};

// =============================================================================
// THEME TOGGLE
// =============================================================================
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;

    toggle.addEventListener('click', () => {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Add transition class for smooth change
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';

        console.log(`🎨 Theme changed to: ${newTheme}`);
    });
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleDateString('vi-VN') + ' ' + date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

function showLoading(text = 'Đang xử lý...') {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = overlay.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

function showElement(id) {
    document.getElementById(id)?.classList.remove('hidden');
}

function hideElement(id) {
    document.getElementById(id)?.classList.add('hidden');
}

// =============================================================================
// DASHBOARD FUNCTIONS
// =============================================================================
function initDashboard() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const clearBtn = document.getElementById('clear-file');
    const analyzeBtn = document.getElementById('analyze-btn');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    if (!uploadZone) return;

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files);
        }
    });

    // Clear file
    clearBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Analyze button
    analyzeBtn?.addEventListener('click', () => analyzefile());

    // New analysis button
    newAnalysisBtn?.addEventListener('click', () => resetDashboard());

    // Export buttons
    document.getElementById('export-json-btn')?.addEventListener('click', () => exportResult('json'));
    document.getElementById('export-pdf-btn')?.addEventListener('click', () => exportResult('pdf'));

    // Clear batch button
    document.getElementById('clear-batch-btn')?.addEventListener('click', () => {
        batchFiles = [];
        resetDashboard();
    });

    // Heatmap checkbox - show/hide all-frames option
    const heatmapCheckbox = document.getElementById('heatmap-checkbox');
    const allFramesOption = document.getElementById('all-frames-option');
    heatmapCheckbox?.addEventListener('change', (e) => {
        if (e.target.checked) {
            allFramesOption?.classList.remove('hidden');
        } else {
            allFramesOption?.classList.add('hidden');
            // Also uncheck all-frames if heatmap is unchecked
            const allFramesCheckbox = document.getElementById('all-frames-checkbox');
            if (allFramesCheckbox) allFramesCheckbox.checked = false;
        }
    });
}

function handleFileSelect(files) {
    // If single file (from old calls or only 1 selected)
    if (files instanceof File) {
        files = [files];
    }

    // Convert FileList to Array
    const fileList = Array.from(files);

    // Validation
    const validFileTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
    const validFiles = fileList.filter(file => {
        if (file.size > MAX_FILE_SIZE) {
            console.warn(`File ${file.name} quá lớn!`);
            return false;
        }
        if (!validFileTypes.some(t => file.type.includes(t.split('/')[1]))) {
            console.warn(`File ${file.name} không hỗ trợ!`);
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) {
        alert('Không có file hợp lệ!');
        return;
    }

    if (validFiles.length === 1) {
        // Single File Mode
        currentFile = validFiles[0];
        initSingleFileUI(currentFile);
    } else {
        // Batch Mode
        batchFiles = validFiles;
        initBatchUI(batchFiles);
    }

    document.getElementById('analyze-btn').disabled = false;
    // Update button text accordingly
    const analyzeBtn = document.getElementById('analyze-btn');
    if (validFiles.length > 1) {
        analyzeBtn.innerHTML = `
            <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            Phân tích Batch (${validFiles.length} files)
        `;
    } else {
        analyzeBtn.innerHTML = `
            <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            Phân tích
        `;
    }
}

function initSingleFileUI(file) {
    const uploadZone = document.getElementById('upload-zone');
    uploadZone.classList.add('has-file');

    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    document.getElementById('file-icon').innerHTML = file.type.startsWith('video') ?
        '<svg class="icon-svg" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg>' :
        '<svg class="icon-svg" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>';

    showElement('file-preview');
    hideElement('batch-queue-section');
}

function initBatchUI(files) {
    const uploadZone = document.getElementById('upload-zone');
    uploadZone.classList.add('has-file');

    hideElement('file-preview');
    showElement('batch-queue-section');

    document.getElementById('batch-count').textContent = files.length;
    document.getElementById('batch-progress-text').textContent = `0/${files.length}`;
    document.getElementById('batch-progress-percent').textContent = '0%';
    document.getElementById('batch-progress-bar').style.width = '0%';

    // Render list
    const tbody = document.getElementById('batch-queue-tbody');
    tbody.innerHTML = files.map((file, index) => `
        <tr id="batch-item-${index}">
            <td>${file.name}</td>
            <td>${formatFileSize(file.size)}</td>
            <td><span class="status-badge status-pending" id="status-${index}">Wait</span></td>
            <td><span id="result-${index}">-</span></td>
        </tr>
    `).join('');
}

function clearFile() {
    currentFile = null;
    batchFiles = [];
    document.getElementById('file-input').value = '';
    document.getElementById('upload-zone').classList.remove('has-file');
    hideElement('file-preview');
    hideElement('batch-queue-section');
    hideElement('result-section');
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('analyze-btn').innerHTML = `
        <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        Phân tích
    `;
}

async function analyzefile() {
    if (batchFiles.length > 0) {
        await processBatchQueue();
    } else if (currentFile) {
        await processSingleFile(currentFile);
    }
}

async function processSingleFile(file) {
    showLoading(`Đang phân tích ${file.type.startsWith('video') ? 'video' : 'ảnh'}...`);
    try {
        const result = await callPredictAPI(file);
        if (result.success) {
            displayResult(result);
            currentHistoryId = result.history_id;
        } else {
            alert('Lỗi: ' + (result.error || 'Không thể phân tích file'));
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Lỗi kết nối server!');
    } finally {
        hideLoading();
    }
}

async function processBatchQueue() {
    isBatchProcessing = true;
    document.getElementById('analyze-btn').disabled = true;

    let processedCount = 0;

    // Helper to update progress
    const updateProgress = () => {
        processedCount++;
        const percent = Math.round((processedCount / batchFiles.length) * 100);
        document.getElementById('batch-progress-text').textContent = `${processedCount}/${batchFiles.length}`;
        document.getElementById('batch-progress-percent').textContent = `${percent}%`;
        document.getElementById('batch-progress-bar').style.width = `${percent}%`;
    };

    for (let i = 0; i < batchFiles.length; i++) {
        const file = batchFiles[i];

        // Update item status to Processing
        const statusEl = document.getElementById(`status-${i}`);
        statusEl.className = 'status-badge status-processing icon-spin';
        statusEl.textContent = '⟳';

        // Scroll to item
        document.getElementById(`batch-item-${i}`).scrollIntoView({ behavior: 'smooth', block: 'center' });

        try {
            const result = await callPredictAPI(file);

            if (result.success) {
                statusEl.className = 'status-badge status-done';
                statusEl.textContent = 'Done';

                const verdictClass = result.verdict === 'FAKE' ? 'text-fake' : 'text-real';
                const verdictColor = result.verdict === 'FAKE' ? 'var(--color-fake)' : 'var(--color-real)';
                document.getElementById(`result-${i}`).innerHTML =
                    `<span style="color: ${verdictColor}; font-weight: bold;">${result.verdict}</span> (${(result.confidence * 100).toFixed(0)}%)`;
            } else {
                statusEl.className = 'status-badge status-error';
                statusEl.textContent = 'Err';
                document.getElementById(`result-${i}`).textContent = 'Error';
            }
        } catch (error) {
            statusEl.className = 'status-badge status-error';
            statusEl.textContent = 'Err';
            document.getElementById(`result-${i}`).textContent = 'Failed';
        }

        updateProgress();
    }

    isBatchProcessing = false;
    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('analyze-btn').textContent = 'Xong! Phân tích lại';
}

async function callPredictAPI(file) {
    const model = document.getElementById('model-select').value;
    const fileType = file.type.startsWith('video') ? 'video' : 'image';
    const generateHeatmap = document.getElementById('heatmap-checkbox')?.checked || false;
    const allFramesHeatmap = document.getElementById('all-frames-checkbox')?.checked || false;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('save_history', 'true');
    formData.append('generate_heatmap', generateHeatmap ? 'true' : 'false');
    formData.append('all_frames_heatmap', allFramesHeatmap ? 'true' : 'false');

    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData
    });

    return await response.json();
}

function displayResult(result) {
    hideElement('upload-section');
    showElement('result-section');

    const verdict = result.verdict;
    const confidence = result.confidence * 100;
    const probs = result.probabilities || {};

    // Verdict
    const verdictEl = document.getElementById('verdict');
    verdictEl.textContent = verdict;
    verdictEl.className = 'verdict ' + verdict.toLowerCase();

    // Verdict explanation (for videos with hybrid verdict)
    const explanationEl = document.getElementById('verdict-explanation');
    if (result.verdict_explanation && explanationEl) {
        explanationEl.textContent = result.verdict_explanation;
        explanationEl.style.display = 'block';
    } else if (explanationEl) {
        explanationEl.style.display = 'none';
    }

    // Confidence meter
    document.getElementById('confidence-circle').style.setProperty('--confidence', confidence);
    document.getElementById('confidence-value').textContent = confidence.toFixed(1) + '%';

    // Probabilities
    const fakeProb = (probs.FAKE || 0) * 100;
    const realProb = (probs.REAL || 0) * 100;

    document.getElementById('fake-bar').style.width = fakeProb + '%';
    document.getElementById('fake-prob').textContent = fakeProb.toFixed(1) + '%';

    document.getElementById('real-bar').style.width = realProb + '%';
    document.getElementById('real-prob').textContent = realProb.toFixed(1) + '%';

    // Details
    document.getElementById('detail-model').textContent = result.model_used || '-';
    document.getElementById('detail-time').textContent = (result.processing_time || 0).toFixed(2) + 's';
    document.getElementById('detail-type').textContent = result.file_info?.file_type || '-';

    // Heatmap display
    const heatmapSection = document.getElementById('heatmap-section');
    const heatmapImage = document.getElementById('heatmap-image');
    const heatmapExplanation = document.getElementById('heatmap-explanation');

    console.log('📋 Checking heatmap in result:', {
        hasHeatmap: !!result.heatmap,
        hasBase64: result.heatmap?.image_base64 ? 'yes (length: ' + result.heatmap.image_base64.length + ')' : 'no'
    });

    if (result.heatmap && result.heatmap.image_base64) {
        heatmapImage.src = result.heatmap.image_base64;
        heatmapExplanation.textContent = result.heatmap.explanation || '';
        heatmapSection?.classList.remove('hidden');
        console.log('🔥 Heatmap displayed');
    } else {
        heatmapSection?.classList.add('hidden');
        console.log('⚠️ No heatmap in result');
    }

    // Timeline display (for videos)
    const timelineSection = document.getElementById('timeline-section');
    const keyFrameSection = document.getElementById('key-frame-section');
    const allFramesGallery = document.getElementById('all-frames-gallery');

    if (result.timeline && result.timeline.length > 0) {
        renderTimeline(result.timeline, result.stats);
        timelineSection?.classList.remove('hidden');
        console.log('📈 Timeline displayed');

        // Key Frame Heatmap (for videos with heatmap enabled)
        if (result.key_frame_heatmap && result.key_frame_heatmap.image_base64) {
            document.getElementById('key-frame-number').textContent = result.key_frame_heatmap.frame_number;
            document.getElementById('key-frame-score').textContent =
                (result.key_frame_heatmap.fake_score * 100).toFixed(1) + '%';
            document.getElementById('key-frame-image').src = result.key_frame_heatmap.image_base64;
            document.getElementById('key-frame-explanation').textContent =
                result.key_frame_heatmap.explanation || '';
            keyFrameSection?.classList.remove('hidden');
            console.log('🔑 Key frame heatmap displayed for frame', result.key_frame_heatmap.frame_number);
        } else {
            keyFrameSection?.classList.add('hidden');
        }

        // All Frames Gallery (when user selected all frames option)
        if (result.all_frames_heatmaps && result.all_frames_heatmaps.length > 0) {
            initAllFramesGallery(result.all_frames_heatmaps);
            allFramesGallery?.classList.remove('hidden');
            // Hide key frame section when showing all frames
            keyFrameSection?.classList.add('hidden');
            console.log('🎞️ All frames gallery displayed with', result.all_frames_heatmaps.length, 'frames');
        } else {
            allFramesGallery?.classList.add('hidden');
        }
    } else {
        timelineSection?.classList.add('hidden');
        keyFrameSection?.classList.add('hidden');
        allFramesGallery?.classList.add('hidden');
    }
}

// All Frames Gallery State
let allFramesData = [];
let currentFrameIndex = 0;

function initAllFramesGallery(frames) {
    allFramesData = frames;
    currentFrameIndex = 0;

    document.getElementById('total-frames-count').textContent = frames.length;

    // Set up navigation buttons
    document.getElementById('prev-frame-btn')?.addEventListener('click', () => navigateFrame(-1));
    document.getElementById('next-frame-btn')?.addEventListener('click', () => navigateFrame(1));

    // Display first frame
    displayGalleryFrame(0);
}

function navigateFrame(direction) {
    const newIndex = currentFrameIndex + direction;
    if (newIndex >= 0 && newIndex < allFramesData.length) {
        displayGalleryFrame(newIndex);
    }
}

function displayGalleryFrame(index) {
    if (index < 0 || index >= allFramesData.length) return;

    currentFrameIndex = index;
    const frame = allFramesData[index];

    document.getElementById('current-frame-idx').textContent = index + 1;
    document.getElementById('gallery-frame-number').textContent = frame.frame_number;
    document.getElementById('gallery-confidence').textContent = (frame.confidence * 100).toFixed(1) + '%';
    document.getElementById('gallery-frame-image').src = frame.image_base64;

    // Update verdict badge
    const verdictBadge = document.getElementById('gallery-verdict-badge');
    verdictBadge.textContent = frame.verdict;
    verdictBadge.style.color = frame.verdict === 'FAKE' ? 'var(--color-fake)' : 'var(--color-real)';

    // Update button states
    document.getElementById('prev-frame-btn').disabled = index === 0;
    document.getElementById('next-frame-btn').disabled = index === allFramesData.length - 1;
}

// Timeline Chart Instance (for cleanup)
let timelineChartInstance = null;

function renderTimeline(timeline, stats) {
    // Update stats badges
    if (stats) {
        document.getElementById('timeline-frames').textContent = stats.total_frames || timeline.length;
        document.getElementById('timeline-fake-count').textContent = stats.fake_count || 0;
        document.getElementById('timeline-real-count').textContent = stats.real_count || 0;
    }

    // Prepare chart data
    const labels = timeline.map(item => `Frame ${item.frame}`);
    const confidences = timeline.map(item => item.confidence * 100);
    const verdicts = timeline.map(item => item.verdict);

    // Color based on verdict
    const backgroundColors = verdicts.map(v =>
        v === 'FAKE' ? 'rgba(255, 71, 87, 0.6)' : 'rgba(46, 213, 115, 0.6)'
    );
    const borderColors = verdicts.map(v =>
        v === 'FAKE' ? 'rgb(255, 71, 87)' : 'rgb(46, 213, 115)'
    );

    // Destroy previous chart if exists
    if (timelineChartInstance) {
        timelineChartInstance.destroy();
    }

    // Create chart
    const ctx = document.getElementById('timeline-chart').getContext('2d');

    timelineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: confidences,
                borderColor: 'rgb(108, 99, 255)',
                backgroundColor: 'rgba(108, 99, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: backgroundColors,
                pointBorderColor: borderColors,
                pointRadius: 5,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const idx = context.dataIndex;
                            const verdict = verdicts[idx];
                            const conf = confidences[idx].toFixed(1);
                            return `${verdict}: ${conf}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Frames',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxTicksLimit: 10
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    const frame = timeline[idx];
                    alert(`Frame ${frame.frame}: ${frame.verdict} (${(frame.confidence * 100).toFixed(1)}%)`);
                }
            }
        }
    });

    console.log('📈 Timeline chart rendered with', timeline.length, 'points');
}

function resetDashboard() {
    clearFile();
    hideElement('result-section');
    hideElement('heatmap-section');
    hideElement('timeline-section');
    showElement('upload-section');
    currentHistoryId = null;

    // Destroy chart if exists
    if (timelineChartInstance) {
        timelineChartInstance.destroy();
        timelineChartInstance = null;
    }
}

async function exportResult(format) {
    if (!currentHistoryId) {
        alert('Không có kết quả để xuất!');
        return;
    }

    showLoading('Đang tạo file...');

    try {
        const response = await fetch(`${API_BASE}/export/${format}/${currentHistoryId}`);

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `deepfake_report_${currentHistoryId}.${format}`;
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            const result = await response.json();
            alert('Lỗi: ' + (result.error || 'Không thể xuất file'));
        }
    } catch (error) {
        console.error('Export error:', error);
        alert('Lỗi kết nối server!');
    } finally {
        hideLoading();
    }
}

// =============================================================================
// HISTORY PAGE FUNCTIONS
// =============================================================================
function initHistoryPage() {
    loadStatistics();
    loadHistory();

    // Filter button
    document.getElementById('apply-filters-btn')?.addEventListener('click', () => {
        historyPage = 1;
        historyFilters = {
            file_type: document.getElementById('filter-type').value,
            verdict: document.getElementById('filter-verdict').value,
            model: document.getElementById('filter-model').value
        };
        loadHistory();
    });

    // Clear history button
    document.getElementById('clear-history-btn')?.addEventListener('click', () => {
        showDeleteModal('all');
    });

    // Modal buttons
    document.getElementById('cancel-delete-btn')?.addEventListener('click', hideDeleteModal);
    document.getElementById('confirm-delete-btn')?.addEventListener('click', confirmDelete);
}

async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/statistics`);
        const data = await response.json();

        if (data.success) {
            const stats = data.statistics;
            document.getElementById('stat-total').textContent = stats.total_predictions || 0;
            document.getElementById('stat-fake').textContent = stats.fake_count || 0;
            document.getElementById('stat-real').textContent = stats.real_count || 0;
            document.getElementById('stat-accuracy').textContent = ((stats.avg_confidence || 0) * 100).toFixed(1) + '%';
        }
    } catch (error) {
        console.error('Statistics error:', error);
    }
}

async function loadHistory() {
    try {
        const params = new URLSearchParams({
            page: historyPage,
            per_page: 10,
            ...historyFilters
        });

        // Remove empty params
        for (const [key, value] of [...params.entries()]) {
            if (!value) params.delete(key);
        }

        const response = await fetch(`${API_BASE}/history?${params}`);
        const data = await response.json();

        if (data.success) {
            renderHistoryTable(data.items);
            renderPagination(data.page, data.pages);

            // Show/hide empty state
            if (data.items.length === 0) {
                showElement('empty-state');
                hideElement('pagination');
            } else {
                hideElement('empty-state');
                showElement('pagination');
            }
        }
    } catch (error) {
        console.error('History error:', error);
    }
}

function renderHistoryTable(items) {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    // Filter out items without valid numeric id FIRST
    const validItems = items.filter(item => {
        const isValid = Number.isInteger(item.id) && item.id > 0;
        if (!isValid) {
            console.warn('⚠️ Skipping item with invalid id:', item);
        }
        return isValid;
    });

    tbody.innerHTML = validItems.map(item => `
        <tr data-id="${item.id}">
            <td>${formatDate(item.created_at)}</td>
            <td>${item.file_name || '-'}</td>
            <td><span class="badge badge-${item.file_type}">${item.file_type}</span></td>
            <td>${item.model_used || '-'}</td>
            <td><span class="badge badge-${item.verdict?.toLowerCase()}">${item.verdict}</span></td>
            <td>${((item.confidence || 0) * 100).toFixed(1)}%</td>
            <td>
                <button class="btn btn-secondary btn-sm action-export-json" data-id="${item.id}" title="Xuất JSON">
                    <svg class="icon-svg sm" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                </button>
                <button class="btn btn-secondary btn-sm action-export-pdf" data-id="${item.id}" title="Xuất PDF">
                    <svg class="icon-svg sm" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" y1="18" x2="12" y2="12"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>
                </button>
                <button class="btn btn-danger btn-sm action-delete" data-id="${item.id}" title="Xóa">
                    <svg class="icon-svg sm" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            </td>
        </tr>
    `).join('');

    // Use event delegation for actions - prevents null id issues
    tbody.onclick = function (e) {
        const btn = e.target.closest('button');
        if (!btn) return;

        const id = parseInt(btn.dataset.id, 10);
        if (!Number.isInteger(id) || id <= 0) {
            console.error('❌ Invalid id from button:', btn.dataset.id);
            alert('Lỗi: ID không hợp lệ!');
            return;
        }

        if (btn.classList.contains('action-export-json')) {
            exportHistoryItem(id, 'json');
        } else if (btn.classList.contains('action-export-pdf')) {
            exportHistoryItem(id, 'pdf');
        } else if (btn.classList.contains('action-delete')) {
            showDeleteModal(id);
        }
    };
}

function renderPagination(current, total) {
    const pagination = document.getElementById('pagination');
    if (!pagination || total <= 1) {
        if (pagination) pagination.innerHTML = '';
        return;
    }

    let html = '';

    // Previous button
    if (current > 1) {
        html += `<button class="pagination-btn" onclick="goToPage(${current - 1})">←</button>`;
    }

    // Page numbers
    for (let i = 1; i <= total; i++) {
        if (i === current) {
            html += `<button class="pagination-btn active">${i}</button>`;
        } else if (i === 1 || i === total || (i >= current - 2 && i <= current + 2)) {
            html += `<button class="pagination-btn" onclick="goToPage(${i})">${i}</button>`;
        } else if (i === current - 3 || i === current + 3) {
            html += `<span class="pagination-btn">...</span>`;
        }
    }

    // Next button
    if (current < total) {
        html += `<button class="pagination-btn" onclick="goToPage(${current + 1})">→</button>`;
    }

    pagination.innerHTML = html;
}

function goToPage(page) {
    historyPage = page;
    loadHistory();
}

async function exportHistoryItem(id, format) {
    showLoading('Đang tạo file...');

    try {
        const response = await fetch(`${API_BASE}/export/${format}/${id}`);

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `deepfake_report_${id}.${format}`;
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            const result = await response.json();
            alert('Lỗi: ' + (result.error || 'Không thể xuất file'));
        }
    } catch (error) {
        console.error('Export error:', error);
        alert('Lỗi kết nối server!');
    } finally {
        hideLoading();
    }
}

// Delete modal
let deleteTarget = null;

function showDeleteModal(target) {
    // Validate target - prevent null/undefined
    if (target === null || target === undefined) {
        console.error('showDeleteModal called with invalid target:', target);
        alert('Lỗi: Không thể xác định mục cần xóa');
        return;
    }

    deleteTarget = target;
    const modal = document.getElementById('delete-modal');
    const message = document.getElementById('delete-message');

    if (target === 'all') {
        message.textContent = 'Bạn có chắc chắn muốn xóa TẤT CẢ lịch sử?';
    } else {
        message.textContent = 'Bạn có chắc chắn muốn xóa mục này?';
    }

    modal.classList.add('active');
}

function hideDeleteModal() {
    document.getElementById('delete-modal').classList.remove('active');
    deleteTarget = null;
}

async function confirmDelete() {
    if (!deleteTarget) {
        console.error('❌ confirmDelete called but deleteTarget is null');
        return;
    }

    // CRITICAL: Save target BEFORE hiding modal (which resets deleteTarget to null)
    const targetToDelete = deleteTarget;

    hideDeleteModal();
    showLoading('Đang xóa...');

    try {
        let response;

        if (targetToDelete === 'all') {
            response = await fetch(`${API_BASE}/history`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confirm: true })
            });
        } else {
            console.log('🗑️ Deleting item with id:', targetToDelete);
            response = await fetch(`${API_BASE}/history/${targetToDelete}`, {
                method: 'DELETE'
            });
        }

        const result = await response.json();

        if (result.success) {
            loadStatistics();
            loadHistory();
        } else {
            alert('Lỗi: ' + (result.error || 'Không thể xóa'));
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Lỗi kết nối server!');
    } finally {
        hideLoading();
    }
}

// =============================================================================
// GLOBAL INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 DeepFake Detector V2.0 loaded');

    // Initialize theme toggle on all pages
    initThemeToggle();
});
