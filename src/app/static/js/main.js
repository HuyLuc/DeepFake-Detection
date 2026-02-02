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
            handleFileSelect(e.target.files[0]);
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
            handleFileSelect(e.dataTransfer.files[0]);
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
}

function handleFileSelect(file) {
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        alert('File quá lớn! Giới hạn: 100MB');
        return;
    }

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
    if (!validTypes.some(t => file.type.includes(t.split('/')[1]))) {
        alert('Loại file không được hỗ trợ!');
        return;
    }

    currentFile = file;

    // Update UI
    const uploadZone = document.getElementById('upload-zone');
    uploadZone.classList.add('has-file');

    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    document.getElementById('file-icon').textContent = file.type.startsWith('video') ? '🎬' : '🖼️';

    showElement('file-preview');
    document.getElementById('analyze-btn').disabled = false;
}

function clearFile() {
    currentFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('upload-zone').classList.remove('has-file');
    hideElement('file-preview');
    document.getElementById('analyze-btn').disabled = true;
}

async function analyzefile() {
    if (!currentFile) return;

    const model = document.getElementById('model-select').value;
    const fileType = currentFile.type.startsWith('video') ? 'video' : 'image';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model', model);
    formData.append('save_history', 'true');

    showLoading(`Đang phân tích ${fileType === 'video' ? 'video' : 'ảnh'}...`);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

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
}

function resetDashboard() {
    clearFile();
    hideElement('result-section');
    showElement('upload-section');
    currentHistoryId = null;
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

    tbody.innerHTML = items.map(item => `
        <tr data-id="${item.id}">
            <td>${formatDate(item.created_at)}</td>
            <td>${item.file_name || '-'}</td>
            <td><span class="badge badge-${item.file_type}">${item.file_type}</span></td>
            <td>${item.model_used || '-'}</td>
            <td><span class="badge badge-${item.verdict?.toLowerCase()}">${item.verdict}</span></td>
            <td>${((item.confidence || 0) * 100).toFixed(1)}%</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="exportHistoryItem(${item.id}, 'json')">📄</button>
                <button class="btn btn-secondary btn-sm" onclick="exportHistoryItem(${item.id}, 'pdf')">📑</button>
                <button class="btn btn-danger btn-sm" onclick="showDeleteModal(${item.id})">🗑️</button>
            </td>
        </tr>
    `).join('');
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
    if (!deleteTarget) return;

    hideDeleteModal();
    showLoading('Đang xóa...');

    try {
        let response;

        if (deleteTarget === 'all') {
            response = await fetch(`${API_BASE}/history`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confirm: true })
            });
        } else {
            response = await fetch(`${API_BASE}/history/${deleteTarget}`, {
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
