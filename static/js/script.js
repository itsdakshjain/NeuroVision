// NeuroScan AI - Interactive JavaScript

// ==================== Configuration ====================
const API_BASE_URL = 'http://localhost:5000';

// ==================== State Management ====================
let uploadedFile = null;
let currentResults = null;
let isAnalyzing = false;
let loadingProgressInterval = null;
const analyzeButtonIdleHTML = `
    <span class="btn-content">
        <i class="fas fa-microscope"></i>
        <span>Begin Analysis</span>
    </span>
    <span class="btn-shine"></span>
`;
const analyzeButtonLoadingHTML = `
    <span class="btn-content">
        <i class="fas fa-spinner fa-spin"></i>
        <span>Analyzing...</span>
    </span>
    <span class="btn-shine"></span>
`;

// ==================== DOM Elements ====================
const elements = {
    // Upload elements
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    imagePreview: document.getElementById('imagePreview'),
    previewImage: document.getElementById('previewImage'),
    removeBtn: document.getElementById('removeBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    // Card states (V2 - new design)
    uploadCard: document.getElementById('uploadCard'),
    loadingCard: document.getElementById('loadingCard'),
    resultsContainer: document.getElementById('resultsContainer'),
    
    // Loading elements (V2)
    progressFill: document.getElementById('progressFill'),
    progressPercent: document.getElementById('progressPercent'),
    
    // Consultation progress container
    consultationProgress: document.getElementById('consultationProgress'),
    
    // Doctor review elements (V2)
    doctorScanPreview: document.getElementById('doctorScanPreview'),
    thoughtBubble: document.getElementById('thoughtBubble'),
    doctorThought: document.getElementById('doctorThought'),
    
    // Analysis checklist items
    checkItem1: document.getElementById('checkItem1'),
    checkItem2: document.getElementById('checkItem2'),
    checkItem3: document.getElementById('checkItem3'),
    checkItem4: document.getElementById('checkItem4'),
    
    // AI model indicators
    aiModel1: document.getElementById('aiModel1'),
    aiModel2: document.getElementById('aiModel2'),
    
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn')
};

// Doctor thoughts for each stage
const doctorThoughts = [
    "Let me carefully examine this MRI scan...",
    "I'm checking the scan quality and resolution. Looks good!",
    "Now examining different brain regions for any abnormalities...",
    "Analyzing the tissue density and contrast patterns...",
    "Looking closely at potential areas of concern...",
    "Cross-referencing with our AI analysis...",
    "Almost done with my review. Preparing the diagnosis..."
];

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeAnimations();
    initializeAccordion();
    initializeConsultationProgress();
    checkAPIHealth();
});

// Initialize consultation progress on page load
function initializeConsultationProgress() {
    // Start at step 1 (Check-In)
    updateConsultationProgress(1);
}

// ==================== Accordion Functionality ====================
function initializeAccordion() {
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    accordionHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const accordionItem = this.parentElement;
            const content = accordionItem.querySelector('.accordion-content');
            const icon = this.querySelector('.fa-chevron-down');
            const isActive = accordionItem.classList.contains('active');
            
            // Close all other accordion items
            document.querySelectorAll('.accordion-item').forEach(item => {
                if (item !== accordionItem) {
                    item.classList.remove('active');
                    const otherContent = item.querySelector('.accordion-content');
                    const otherIcon = item.querySelector('.fa-chevron-down');
                    if (otherContent) {
                        otherContent.style.maxHeight = null;
                        otherContent.style.opacity = '0';
                    }
                    if (otherIcon) {
                        otherIcon.style.transform = 'rotate(0deg)';
                    }
                }
            });
            
            // Toggle current accordion item
            if (!isActive) {
                accordionItem.classList.add('active');
                if (content) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                    content.style.opacity = '1';
                }
                if (icon) {
                    icon.style.transform = 'rotate(180deg)';
                }
            } else {
                accordionItem.classList.remove('active');
                if (content) {
                    content.style.maxHeight = null;
                    content.style.opacity = '0';
                }
                if (icon) {
                    icon.style.transform = 'rotate(0deg)';
                }
            }
        });
    });
}

// ==================== Event Listeners ====================
function initializeEventListeners() {
    // Upload area click
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });
    
    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Remove button
    elements.removeBtn.addEventListener('click', handleRemoveImage);
    
    // Analyze button
    elements.analyzeBtn.addEventListener('click', handleAnalyze);
    
    // Navigation
    elements.navLinks.forEach(link => {
        link.addEventListener('click', handleNavClick);
    });
    
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', handleSmoothScroll);
    });
    
    // Scroll spy for navigation
    window.addEventListener('scroll', handleScrollSpy);
}

// ==================== File Handling ====================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadArea.style.borderColor = 'var(--planetary)';
    elements.uploadArea.style.backgroundColor = 'var(--sky)';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadArea.style.borderColor = 'var(--universe)';
    elements.uploadArea.style.backgroundColor = 'var(--gray-50)';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    handleDragLeave(e);
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function validateAndPreviewFile(file) {
    if (isAnalyzing) {
        showNotification('Analysis is already in progress. Please wait for it to finish.', 'info');
        return;
    }

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'];
    
    const fileName = file.name.toLowerCase();
    const isValidType = validTypes.includes(file.type) || 
                       validExtensions.some(ext => fileName.endsWith(ext));
    
    if (!isValidType) {
        showNotification('Please upload a valid image file (JPG, PNG, TIF, TIFF)', 'error');
        return;
    }
    
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size must be less than 16MB', 'error');
        return;
    }
    
    // Store file and preview
    uploadedFile = file;
    previewImage(file);
}

function previewImage(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadArea.style.display = 'none';
        elements.imagePreview.style.display = 'block';
        setAnalyzeControlsLoading(false);
        
        // Animate preview
        elements.imagePreview.classList.add('fade-in');
        
        // Update progress to step 2 (Upload Scan - completed uploading)
        updateConsultationProgress(2);
    };
    
    reader.readAsDataURL(file);
}

function handleRemoveImage(e) {
    e.stopPropagation();

    if (isAnalyzing) {
        showNotification('Please wait until the current analysis is complete.', 'info');
        return;
    }
    
    uploadedFile = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    elements.uploadArea.style.display = 'block';
    elements.imagePreview.style.display = 'none';
    elements.analyzeBtn.disabled = true;
    
    // Reset progress back to step 1 (Check-In)
    updateConsultationProgress(1);
}

// ==================== Analysis ====================
async function handleAnalyze() {
    if (isAnalyzing) {
        showNotification('Analysis is already in progress. Please wait for the result.', 'info');
        return;
    }

    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }

    isAnalyzing = true;
    setAnalyzeControlsLoading(true);
    
    // Show loading state
    showLoadingState();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    try {
        // Make API request
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let errorMessage = 'Analysis failed';
            try {
                const error = await response.json();
                errorMessage = error.error || errorMessage;
            } catch {
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const results = await response.json();
        currentResults = results;
        
        // Show results
        setTimeout(() => {
            showResults(results);
        }, 1000);
        
    } catch (error) {
        console.error('Analysis error:', error);
        isAnalyzing = false;
        hideLoadingState();
        showNotification(error.message || 'Failed to analyze image. Please try again.', 'error');
    }
}

function setAnalyzeControlsLoading(isLoading) {
    if (elements.analyzeBtn) {
        elements.analyzeBtn.disabled = isLoading || !uploadedFile;
        elements.analyzeBtn.setAttribute('aria-busy', String(isLoading));
        elements.analyzeBtn.innerHTML = isLoading
            ? analyzeButtonLoadingHTML
            : analyzeButtonIdleHTML;
    }

    if (elements.removeBtn) {
        elements.removeBtn.disabled = isLoading;
        elements.removeBtn.setAttribute('aria-disabled', String(isLoading));
    }
}

function showLoadingState() {
    elements.uploadCard.style.display = 'none';
    elements.loadingCard.style.display = 'block';
    elements.resultsContainer.style.display = 'none';
    
    // Update consultation progress - move to step 3 (Doctor Review)
    updateConsultationProgress(3);
    
    // Set the doctor's scan preview to the uploaded image
    if (uploadedFile && elements.doctorScanPreview) {
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.doctorScanPreview.src = e.target.result;
        };
        reader.readAsDataURL(uploadedFile);
    }
    
    // Start doctor review animation
    startDoctorReviewAnimation();
    
    // Animate progress
    let progress = 0;
    if (loadingProgressInterval) {
        clearInterval(loadingProgressInterval);
    }

    loadingProgressInterval = setInterval(() => {
        progress += 2;
        const currentProgress = Math.min(progress, 90);
        
        if (elements.progressFill) {
            elements.progressFill.style.width = `${currentProgress}%`;
        }
        if (elements.progressPercent) {
            elements.progressPercent.textContent = `${Math.round(currentProgress)}%`;
        }
        
        // Update checklist items based on progress
        if (progress >= 20) {
            updateChecklistItem(elements.checkItem1, 'completed');
            updateChecklistItem(elements.checkItem2, 'active');
        }
        if (progress >= 45) {
            updateChecklistItem(elements.checkItem2, 'completed');
            updateChecklistItem(elements.checkItem3, 'active');
        }
        if (progress >= 70) {
            updateChecklistItem(elements.checkItem3, 'completed');
            updateChecklistItem(elements.checkItem4, 'active');
            // Update AI models
            updateAIModel(elements.aiModel1, 'completed');
            updateAIModel(elements.aiModel2, 'active');
        }
        
        if (progress >= 90) {
            clearInterval(loadingProgressInterval);
            loadingProgressInterval = null;
        }
    }, 50);
}

// Update consultation progress steps
function updateConsultationProgress(activeStep) {
    if (!elements.consultationProgress) return;
    
    const steps = elements.consultationProgress.querySelectorAll('.progress-step');
    
    steps.forEach((step) => {
        const stepNum = parseInt(step.getAttribute('data-step'));
        step.classList.remove('active', 'completed');
        
        if (stepNum < activeStep) {
            step.classList.add('completed');
        } else if (stepNum === activeStep) {
            step.classList.add('active');
        }
    });
}

// Update checklist item state
function updateChecklistItem(item, state) {
    if (item) {
        item.classList.remove('active', 'completed');
        if (state) {
            item.classList.add(state);
        }
    }
}

// Update AI model indicator
function updateAIModel(model, state) {
    if (model) {
        model.classList.remove('active', 'completed');
        if (state) {
            model.classList.add(state);
        }
    }
}

// Doctor review animation controller
let doctorAnimationInterval = null;
let thoughtIndex = 0;

function startDoctorReviewAnimation() {
    thoughtIndex = 0;
    
    // Initialize checklist items
    updateChecklistItem(elements.checkItem1, 'active');
    updateChecklistItem(elements.checkItem2, null);
    updateChecklistItem(elements.checkItem3, null);
    updateChecklistItem(elements.checkItem4, null);
    
    // Initialize AI models
    updateAIModel(elements.aiModel1, 'active');
    updateAIModel(elements.aiModel2, null);
    
    // Update doctor thought
    if (elements.doctorThought) {
        elements.doctorThought.textContent = doctorThoughts[0];
    }
    
    // Start the animation cycle
    doctorAnimationInterval = setInterval(() => {
        // Update thoughts
        thoughtIndex = (thoughtIndex + 1) % doctorThoughts.length;
        if (elements.doctorThought && elements.thoughtBubble) {
            // Fade out
            elements.thoughtBubble.style.opacity = '0.5';
            setTimeout(() => {
                elements.doctorThought.textContent = doctorThoughts[thoughtIndex];
                elements.thoughtBubble.style.opacity = '1';
            }, 200);
        }
    }, 1500);
}

function stopDoctorReviewAnimation() {
    if (doctorAnimationInterval) {
        clearInterval(doctorAnimationInterval);
        doctorAnimationInterval = null;
    }
    
    // Mark all checklist items as completed
    updateChecklistItem(elements.checkItem1, 'completed');
    updateChecklistItem(elements.checkItem2, 'completed');
    updateChecklistItem(elements.checkItem3, 'completed');
    updateChecklistItem(elements.checkItem4, 'completed');
    
    // Mark all AI models as completed
    updateAIModel(elements.aiModel1, 'completed');
    updateAIModel(elements.aiModel2, 'completed');
    
    // Final thought
    if (elements.doctorThought) {
        elements.doctorThought.textContent = "Analysis complete! Here are the results...";
    }
}

function hideLoadingState() {
    // Stop doctor review animation
    stopDoctorReviewAnimation();

    if (loadingProgressInterval) {
        clearInterval(loadingProgressInterval);
        loadingProgressInterval = null;
    }
    
    elements.uploadCard.style.display = 'block';
    elements.loadingCard.style.display = 'none';
    setAnalyzeControlsLoading(false);
    
    // Reset loading state
    if (elements.progressFill) {
        elements.progressFill.style.width = '0%';
    }
    if (elements.progressPercent) {
        elements.progressPercent.textContent = '0%';
    }
    
    // Reset consultation progress - if file still uploaded, show step 2, otherwise step 1
    updateConsultationProgress(uploadedFile ? 2 : 1);
    
    // Reset checklist items
    updateChecklistItem(elements.checkItem1, null);
    updateChecklistItem(elements.checkItem2, null);
    updateChecklistItem(elements.checkItem3, null);
    updateChecklistItem(elements.checkItem4, null);
    
    // Reset AI models
    updateAIModel(elements.aiModel1, null);
    updateAIModel(elements.aiModel2, null);
}

function showResults(results) {
    // Stop doctor animation
    stopDoctorReviewAnimation();

    if (loadingProgressInterval) {
        clearInterval(loadingProgressInterval);
        loadingProgressInterval = null;
    }
    
    // Complete progress
    if (elements.progressFill) {
        elements.progressFill.style.width = '100%';
    }
    if (elements.progressPercent) {
        elements.progressPercent.textContent = '100%';
    }
    
    // Update all checklist items to completed
    updateChecklistItem(elements.checkItem1, 'completed');
    updateChecklistItem(elements.checkItem2, 'completed');
    updateChecklistItem(elements.checkItem3, 'completed');
    updateChecklistItem(elements.checkItem4, 'completed');
    
    // Update AI models to completed
    updateAIModel(elements.aiModel1, 'completed');
    updateAIModel(elements.aiModel2, 'completed');
    
    // Update consultation progress to final step (Report)
    updateConsultationProgress(4);
    
    // Hide loading, show results
    setTimeout(() => {
        elements.loadingCard.style.display = 'none';
        elements.resultsContainer.style.display = 'block';
        
        // Generate results HTML
        const resultsHTML = generateResultsHTML(results);
        elements.resultsContainer.innerHTML = resultsHTML;
        
        // Animate results
        elements.resultsContainer.classList.add('fade-in');
        
        // Add event listener to new scan button
        const newScanBtn = document.getElementById('newScanBtn');
        if (newScanBtn) {
            newScanBtn.addEventListener('click', handleNewScan);
        }
        
        // Scroll to results
        elements.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        isAnalyzing = false;
        setAnalyzeControlsLoading(false);
        
    }, 500);
}

function generateResultsHTML(results) {
    const hasTumor = results.has_tumor;
    const confidence = (results.confidence * 100).toFixed(1);
    const report = results.detailed_report || {};
    
    // Doctor's verdict based on results
    const doctorVerdict = hasTumor 
        ? `Based on my examination and the AI analysis, I've identified an abnormal tissue mass in the MRI scan. The ${confidence}% confidence level suggests this requires immediate attention. I recommend scheduling a follow-up consultation with a neurologist for further evaluation and to discuss treatment options.`
        : `After carefully reviewing this MRI scan along with our AI analysis, I'm pleased to report that no tumor indicators were detected. The brain tissue appears healthy with normal density patterns. However, please continue with regular check-ups as recommended by your primary physician.`;
    
    let html = `
        <!-- Share & Actions Bar -->
        <div class="report-actions-bar">
            <div class="report-id">
                <i class="fas fa-file-medical-alt"></i>
                <span>Report ID: ${report.scan_id || 'N/A'}</span>
            </div>
            <div class="action-buttons">
                <button class="action-btn" type="button" onclick="shareReport('copy')" title="Copy Link">
                    <i class="fas fa-link"></i>
                    <span>Copy Link</span>
                </button>
                <button class="action-btn" type="button" onclick="shareReport('download')" title="Download Report">
                    <i class="fas fa-download"></i>
                    <span>Download</span>
                </button>
                <button class="action-btn" type="button" onclick="shareReport('email')" title="Email Report">
                    <i class="fas fa-envelope"></i>
                    <span>Email</span>
                </button>
                <button class="action-btn save-btn" type="button" onclick="saveToHistory()" title="Save to History">
                    <i class="fas fa-bookmark"></i>
                    <span>Save</span>
                </button>
            </div>
        </div>
        
        <!-- Doctor's Verdict Section -->
        <div class="doctor-verdict-section">
            <div class="verdict-header">
                <div class="verdict-doctor">
                    <div class="verdict-avatar">
                        <i class="fas fa-user-md"></i>
                    </div>
                    <div class="verdict-info">
                        <span class="verdict-name">Dr. Sarah Mitchell</span>
                        <span class="verdict-title">Senior Neuroradiologist</span>
                    </div>
                </div>
                <div class="verdict-stamp ${hasTumor ? 'concern' : 'clear'}">
                    <i class="fas ${hasTumor ? 'fa-exclamation-circle' : 'fa-check-circle'}"></i>
                    <span>${hasTumor ? 'Requires Attention' : 'All Clear'}</span>
                </div>
            </div>
            <div class="verdict-content">
                <i class="fas fa-quote-left"></i>
                <p>${doctorVerdict}</p>
            </div>
            <div class="verdict-signature">
                <div class="signature-line">
                    <span class="signature">Dr. S. Mitchell</span>
                    <span class="date">${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
                </div>
            </div>
        </div>
        
        <div class="results-header">
            <div class="result-badge ${hasTumor ? 'positive' : 'negative'}">
                <i class="fas ${hasTumor ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                <span>${hasTumor ? 'Tumor Detected' : 'No Tumor Detected'}</span>
            </div>
            <div class="confidence-score">${confidence}%</div>
            <div class="confidence-label">AI Confidence Score</div>
        </div>
        
        <div class="results-grid">
            <div class="result-image-card">
                <img src="${results.original_image}" alt="Original MRI">
                <div class="result-image-label">Original MRI Scan</div>
            </div>
    `;
    
    if (hasTumor && results.segmentation) {
        html += `
            <div class="result-image-card">
                <img src="${results.segmentation.mask}" alt="Tumor Mask">
                <div class="result-image-label">AI-Generated Mask</div>
            </div>
            
            <div class="result-image-card">
                <img src="${results.segmentation.overlay}" alt="Tumor Overlay">
                <div class="result-image-label">Tumor Localization</div>
            </div>
        `;
    }
    
    html += `</div>`;
    
    // Detailed Report Section
    if (report.tumor_characteristics) {
        const chars = report.tumor_characteristics;
        const severity = report.severity_details || {};
        
        html += `
            <div class="detailed-report-section">
                <h3 class="report-section-title">
                    <i class="fas fa-file-medical"></i>
                    Comprehensive Diagnostic Report
                </h3>
                
                <!-- Report Summary Cards -->
                <div class="report-summary-grid">
                    <div class="summary-card ${hasTumor ? 'alert' : 'success'}">
                        <div class="summary-icon">
                            <i class="fas ${hasTumor ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                        </div>
                        <div class="summary-content">
                            <span class="summary-label">Detection Status</span>
                            <span class="summary-value">${hasTumor ? 'Abnormality Detected' : 'No Abnormality'}</span>
                        </div>
                    </div>
                    
                    <div class="summary-card">
                        <div class="summary-icon">
                            <i class="fas fa-percentage"></i>
                        </div>
                        <div class="summary-content">
                            <span class="summary-label">Confidence Score</span>
                            <span class="summary-value">${chars.confidence_score || confidence + '%'}</span>
                        </div>
                    </div>
                    
                    ${hasTumor ? `
                    <div class="summary-card" style="border-left: 4px solid ${severity.color || '#f59e0b'}">
                        <div class="summary-icon" style="color: ${severity.color || '#f59e0b'}">
                            <i class="fas fa-heartbeat"></i>
                        </div>
                        <div class="summary-content">
                            <span class="summary-label">Severity Level</span>
                            <span class="summary-value" style="color: ${severity.color || '#f59e0b'}">${severity.level || 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="summary-card">
                        <div class="summary-icon">
                            <i class="fas fa-clock"></i>
                        </div>
                        <div class="summary-content">
                            <span class="summary-label">Urgency</span>
                            <span class="summary-value">${severity.urgency || 'N/A'}</span>
                        </div>
                    </div>
                    ` : ''}
                </div>
        `;
        
        if (hasTumor && chars.detected) {
            const size = chars.estimated_size || {};
            
            html += `
                <!-- Tumor Characteristics -->
                <div class="report-details-grid">
                    <div class="details-card">
                        <h4><i class="fas fa-ruler-combined"></i> Size & Coverage</h4>
                        <div class="details-list">
                            <div class="detail-row">
                                <span class="detail-label">Coverage Area</span>
                                <span class="detail-value">${chars.coverage_percentage || '0%'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Affected Pixels</span>
                                <span class="detail-value">${chars.affected_pixels || '0'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Est. Width</span>
                                <span class="detail-value">${size.width_mm || '0'} mm</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Est. Height</span>
                                <span class="detail-value">${size.height_mm || '0'} mm</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Est. Area</span>
                                <span class="detail-value">${size.area_mm2 || '0'} mm²</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-card">
                        <h4><i class="fas fa-map-marker-alt"></i> Location & Shape</h4>
                        <div class="details-list">
                            <div class="detail-row">
                                <span class="detail-label">Location</span>
                                <span class="detail-value">${chars.location || 'N/A'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Position Risk</span>
                                <span class="detail-value small">${chars.location_risk || 'N/A'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Shape</span>
                                <span class="detail-value">${chars.shape_assessment || 'N/A'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Mask Confidence</span>
                                <span class="detail-value">${report.mask_confidence || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="details-card full-width">
                        <h4><i class="fas fa-clipboard-list"></i> Clinical Recommendations</h4>
                        <ul class="recommendations-list">
                            ${(report.recommendations || []).map(rec => `
                                <li><i class="fas fa-chevron-right"></i> ${rec}</li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            `;
        } else {
            // No tumor recommendations
            html += `
                <div class="report-details-grid">
                    <div class="details-card full-width success-card">
                        <h4><i class="fas fa-clipboard-list"></i> Recommendations</h4>
                        <ul class="recommendations-list">
                            ${(report.recommendations || []).map(rec => `
                                <li><i class="fas fa-check"></i> ${rec}</li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            `;
        }
        
        // Analysis Metadata
        const metadata = report.analysis_metadata || {};
        html += `
                <div class="analysis-metadata">
                    <h4><i class="fas fa-cog"></i> Analysis Details</h4>
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <span class="meta-label">Models Used</span>
                            <span class="meta-value">${(metadata.models_used || []).join(', ') || 'N/A'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="meta-label">TTA Enabled</span>
                            <span class="meta-value">${metadata.tta_enabled ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="meta-label">Ensemble</span>
                            <span class="meta-value">${metadata.ensemble_enabled ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="meta-label">AI Version</span>
                            <span class="meta-value">${metadata.ai_version || '2.0.0'}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `
        <div class="disclaimer-note">
            <i class="fas fa-info-circle"></i>
            <p><strong>Important:</strong> ${report.disclaimer || 'This analysis is for informational purposes only and should not replace professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment.'}</p>
        </div>
        
        <div class="results-button-group">
            <button class="btn btn-secondary" type="button" onclick="printReport()">
                <i class="fas fa-print"></i>
                Print Report
            </button>
            <button class="btn btn-primary btn-new-scan" id="newScanBtn" type="button">
                <i class="fas fa-plus"></i>
                Analyze Another Scan
            </button>
        </div>
    `;
    
    return html;
}

// ==================== Share & Save Functions ====================
async function shareReport(method) {
    const shareUrl = window.location.origin + '/share/' + (currentResults?.detailed_report?.scan_id || 'demo');
    
    switch(method) {
        case 'copy':
            try {
                await navigator.clipboard.writeText(shareUrl);
                showNotification('Link copied to clipboard!', 'success');
            } catch (err) {
                // Fallback for older browsers
                const input = document.createElement('input');
                input.value = shareUrl;
                document.body.appendChild(input);
                input.select();
                document.execCommand('copy');
                document.body.removeChild(input);
                showNotification('Link copied to clipboard!', 'success');
            }
            break;
            
        case 'download':
            downloadReportAsPDF();
            break;
            
        case 'email':
            const subject = encodeURIComponent('NeuroScan AI - Brain MRI Analysis Report');
            const body = encodeURIComponent(`Please find my brain MRI analysis report from NeuroScan AI:\n\n${shareUrl}\n\nThis report was generated using advanced AI-powered analysis.`);
            window.open(`mailto:?subject=${subject}&body=${body}`);
            break;
    }
}

function downloadReportAsPDF() {
    // Create a printable version and trigger print
    const resultsContent = document.getElementById('resultsContainer').innerHTML;
    const printWindow = window.open('', '_blank');

    if (!printWindow) {
        showNotification('Please allow pop-ups to download or print the report.', 'error');
        return;
    }
    
    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuroScan AI - Medical Report</title>
            <link rel="stylesheet" href="/static/css/style.css">
            <style>
                body { 
                    padding: 20px; 
                    font-family: Arial, sans-serif;
                    background: white;
                }
                .report-actions-bar { display: none; }
                .results-button-group { display: none; }
                @media print {
                    .no-print { display: none; }
                }
            </style>
        </head>
        <body>
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #1e3a5f;">🧠 NeuroScan AI</h1>
                <p>Brain Tumor Detection Report</p>
                <hr>
            </div>
            ${resultsContent}
            <div style="margin-top: 40px; text-align: center; color: #666; font-size: 12px;">
                <p>Generated by NeuroScan AI on ${new Date().toLocaleString()}</p>
                <p>This report is for informational purposes only.</p>
            </div>
        </body>
        </html>
    `);
    
    printWindow.document.close();
    setTimeout(() => {
        printWindow.print();
    }, 500);
}

function printReport() {
    downloadReportAsPDF();
}

async function saveToHistory() {
    if (!currentResults) {
        showNotification('No scan results to save', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/history/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify(currentResults)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Scan saved to your history!', 'success');
            // Update the share URL with the actual token
            if (data.share_token) {
                currentResults.share_token = data.share_token;
            }
        } else {
            if (data.authenticated === false) {
                showNotification('Please login to save scans to history', 'info');
                showAuthModal('login');
            } else {
                showNotification(data.error || 'Failed to save scan', 'error');
            }
        }
    } catch (error) {
        console.error('Save error:', error);
        showNotification('Failed to save scan. Please try again.', 'error');
    }
}

function handleNewScan() {
    isAnalyzing = false;
    setAnalyzeControlsLoading(false);

    // Reset state
    handleRemoveImage({ stopPropagation: () => {} });
    
    // Show upload card
    elements.resultsContainer.style.display = 'none';
    elements.uploadCard.style.display = 'block';
    
    // Reset consultation progress to step 1 (Check-In)
    updateConsultationProgress(1);
    
    // Scroll to upload section
    document.getElementById('scan').scrollIntoView({ behavior: 'smooth' });
}

// ==================== Navigation ====================
function handleNavClick(e) {
    e.preventDefault();
    const clickedLink = e.currentTarget;
    
    // Remove active class from all links
    elements.navLinks.forEach(link => link.classList.remove('active'));
    
    // Add active class to clicked link
    clickedLink.classList.add('active');
    
    // Get target section
    const targetId = clickedLink.getAttribute('href');
    const targetSection = document.querySelector(targetId);
    
    if (targetSection) {
        targetSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function handleSmoothScroll(e) {
    const link = e.target.closest('a[href^="#"]');
    const href = link?.getAttribute('href');
    
    if (href && href.startsWith('#')) {
        e.preventDefault();
        
        // Special handling for #home - scroll to top
        if (href === '#home') {
            smoothScrollTo(0, 1200);
            return;
        }
        
        const targetSection = document.querySelector(href);
        
        if (targetSection) {
            const targetPosition = targetSection.offsetTop - 80; // Account for navbar
            smoothScrollTo(targetPosition, 1000);
        }
    }
}

// Enhanced smooth scroll with custom easing
function smoothScrollTo(targetPosition, duration) {
    const startPosition = window.pageYOffset;
    const distance = targetPosition - startPosition;
    let startTime = null;
    
    function easeOutQuint(t) {
        return 1 - Math.pow(1 - t, 5);
    }
    
    function animation(currentTime) {
        if (startTime === null) startTime = currentTime;
        const timeElapsed = currentTime - startTime;
        const progress = Math.min(timeElapsed / duration, 1);
        const easeProgress = easeOutQuint(progress);
        
        window.scrollTo(0, startPosition + distance * easeProgress);
        
        if (timeElapsed < duration) {
            requestAnimationFrame(animation);
        }
    }
    
    requestAnimationFrame(animation);
}

function handleScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const scrollPosition = window.scrollY + 100;
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            elements.navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${sectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    });
}

// ==================== Animations ====================
function initializeAnimations() {
    // Animate stats on scroll
    const statValues = document.querySelectorAll('.stat-value [data-target], .stat-value[data-target]');
    
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateValue(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    statValues.forEach(stat => observer.observe(stat));
    
    // Initialize features section animations
    initializeFeaturesAnimations();
}

// Features Section Scroll Animations
function initializeFeaturesAnimations() {
    // Animate elements when they come into view
    const animateOnScroll = (entries, observer) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Add delay based on index for staggered effect
                const delay = index * 100;
                setTimeout(() => {
                    entry.target.classList.add('animate-in');
                }, delay);
                observer.unobserve(entry.target);
            }
        });
    };
    
    const scrollObserverOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const scrollObserver = new IntersectionObserver(animateOnScroll, scrollObserverOptions);
    
    // Observe feature highlight card
    const highlightCard = document.querySelector('.feature-highlight-card');
    if (highlightCard) {
        highlightCard.classList.add('animate-ready');
        scrollObserver.observe(highlightCard);
    }
    
    // Observe bento cards
    const bentoCards = document.querySelectorAll('.bento-card');
    bentoCards.forEach(card => {
        card.classList.add('animate-ready');
        scrollObserver.observe(card);
    });
    
    // Observe tech stack items
    const techItems = document.querySelectorAll('.tech-item');
    techItems.forEach(item => {
        item.classList.add('animate-ready');
        scrollObserver.observe(item);
    });
    
    // Observe tech stack card
    const techStackCard = document.querySelector('.tech-stack-card');
    if (techStackCard) {
        techStackCard.classList.add('animate-ready');
        scrollObserver.observe(techStackCard);
    }
    
    // Add CSS for the animations
    const animationStyles = document.createElement('style');
    animationStyles.textContent = `
        .animate-ready {
            opacity: 0;
            transform: translateY(30px);
            transition: opacity 0.6s ease-out, transform 0.6s ease-out;
        }
        
        .animate-ready.animate-in {
            opacity: 1;
            transform: translateY(0);
        }
        
        .feature-highlight-card.animate-ready {
            transform: translateY(40px);
        }
        
        .bento-card.animate-ready {
            transform: translateY(25px) scale(0.98);
            transition: opacity 0.5s ease-out, transform 0.5s ease-out, box-shadow 0.3s ease;
        }
        
        .bento-card.animate-ready.animate-in {
            transform: translateY(0) scale(1);
        }
        
        .tech-item.animate-ready {
            transform: translateY(20px) scale(0.95);
        }
        
        .tech-item.animate-ready.animate-in {
            transform: translateY(0) scale(1);
        }
        
        .tech-stack-card.animate-ready {
            transform: translateY(35px);
        }
        
        /* Pipeline stage hover effect */
        .pipeline-stage .stage-icon {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        }
        
        .pipeline-stage:hover .stage-icon {
            transform: scale(1.15) rotate(5deg);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        /* Bento card icon bounce */
        .bento-card:hover .bento-icon {
            animation: iconBounce 0.5s ease;
        }
        
        @keyframes iconBounce {
            0%, 100% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.15) rotate(-5deg); }
            50% { transform: scale(1.1) rotate(5deg); }
            75% { transform: scale(1.12) rotate(-3deg); }
        }
        
        /* Speed fill animation */
        .speed-fill {
            animation: speedPulse 2s ease-in-out infinite;
        }
        
        @keyframes speedPulse {
            0%, 100% { opacity: 1; width: 85%; }
            50% { opacity: 0.8; width: 80%; }
        }
        
        /* Tumor mark pulse */
        .overlay-tumor-mark {
            animation: tumorPulse 1.5s ease-in-out infinite;
        }
        
        @keyframes tumorPulse {
            0%, 100% { transform: scale(1); opacity: 0.7; box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); }
            50% { transform: scale(1.25); opacity: 1; box-shadow: 0 0 25px rgba(239, 68, 68, 0.7); }
        }
        
        /* Format badge hover */
        .format-badge {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .format-badge:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(51, 78, 172, 0.25);
        }
        
        /* Dataset stat slide */
        .dataset-stat {
            transition: transform 0.3s ease, background 0.3s ease;
        }
        
        /* Tech logo hover glow */
        .tech-logo {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .tech-item:hover .tech-logo {
            transform: scale(1.1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Pipeline connector animation */
        .connector-line-visual {
            position: relative;
            overflow: hidden;
        }
        
        .connector-line-visual::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Bento metric glow on hover */
        .bento-card:hover .bento-metric {
            background: linear-gradient(135deg, rgba(51, 78, 172, 0.1), rgba(112, 152, 209, 0.1));
        }
        
        .bento-card:hover .metric-value {
            text-shadow: 0 0 20px rgba(51, 78, 172, 0.3);
        }
    `;
    document.head.appendChild(animationStyles);
}

function animateValue(element) {
    const target = parseFloat(element.getAttribute('data-target'));
    const duration = 2000;
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        
        // Format based on value type
        if (element.textContent.includes('%')) {
            element.textContent = current.toFixed(1) + '%';
        } else if (target < 10) {
            element.textContent = current.toFixed(1);
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, 16);
}

// ==================== Notifications ====================
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 5rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? 'var(--error)' : type === 'success' ? 'var(--success)' : 'var(--info)'};
        color: white;
        border-radius: 0.75rem;
        box-shadow: var(--shadow-xl);
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
    `;
    
    const icon = type === 'error' ? 'fa-exclamation-circle' : 
                 type === 'success' ? 'fa-check-circle' : 'fa-info-circle';
    
    notification.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ==================== API Health Check ====================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.models_loaded) {
            return;
        } else {
            console.warn('API is running but models may not be loaded');
            showNotification('System initializing. Please wait...', 'info');
        }
    } catch (error) {
        console.error('Failed to connect to API:', error);
        showNotification('Unable to connect to server. Please ensure the Flask app is running.', 'error');
    }
}

// ==================== Utility Functions ====================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==================== Authentication System ====================
let currentUser = null;

async function checkAuthStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/status`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (data.authenticated) {
            currentUser = data.user;
            updateUIForLoggedInUser();
        } else {
            currentUser = null;
            updateUIForLoggedOutUser();
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        currentUser = null;
        updateUIForLoggedOutUser();
    }
}

function updateUIForLoggedInUser() {
    const authButton = document.getElementById('authButton');
    const historyButton = document.getElementById('historyButton');
    
    if (authButton) {
        authButton.innerHTML = `<i class="fas fa-user"></i> ${currentUser.name}`;
        authButton.onclick = showUserMenu;
    }
    
    if (historyButton) {
        historyButton.style.display = 'flex';
    }
}

function updateUIForLoggedOutUser() {
    const authButton = document.getElementById('authButton');
    const historyButton = document.getElementById('historyButton');
    
    if (authButton) {
        authButton.innerHTML = `<i class="fas fa-sign-in-alt"></i> Sign In`;
        authButton.onclick = () => showAuthModal('login');
    }
    
    if (historyButton) {
        historyButton.style.display = 'none';
    }
}

function showAuthModal(mode = 'login') {
    // Create modal if it doesn't exist
    let modal = document.getElementById('authModal');
    if (!modal) {
        modal = createAuthModal();
        document.body.appendChild(modal);
    }
    
    // Switch between login and signup forms
    const loginForm = modal.querySelector('.login-form');
    const signupForm = modal.querySelector('.signup-form');
    const modalTitle = modal.querySelector('.auth-modal-title');
    
    if (mode === 'login') {
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
        modalTitle.textContent = 'Sign In';
    } else {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
        modalTitle.textContent = 'Create Account';
    }
    
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    
    // Focus first input
    setTimeout(() => {
        const firstInput = modal.querySelector(mode === 'login' ? '#loginEmail' : '#signupName');
        if (firstInput) firstInput.focus();
    }, 100);
}

function createAuthModal() {
    const modal = document.createElement('div');
    modal.id = 'authModal';
    modal.className = 'auth-modal';
    modal.innerHTML = `
        <div class="auth-modal-overlay" onclick="closeAuthModal()"></div>
        <div class="auth-modal-content">
            <button class="auth-modal-close" type="button" onclick="closeAuthModal()">
                <i class="fas fa-times"></i>
            </button>
            
            <div class="auth-modal-header">
                <div class="auth-modal-logo">
                    <i class="fas fa-brain"></i>
                </div>
                <h2 class="auth-modal-title">Sign In</h2>
                <p class="auth-modal-subtitle">Access your scan history and more</p>
            </div>
            
            <!-- Login Form -->
            <form class="login-form" onsubmit="handleLogin(event)">
                <div class="auth-input-group">
                    <label for="loginEmail">Email</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-envelope"></i>
                        <input type="email" id="loginEmail" placeholder="Enter your email" required>
                    </div>
                </div>
                
                <div class="auth-input-group">
                    <label for="loginPassword">Password</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="loginPassword" placeholder="Enter your password" required>
                        <button type="button" class="password-toggle" onclick="togglePassword('loginPassword')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                </div>
                
                <button type="submit" class="auth-submit-btn">
                    <span>Sign In</span>
                    <i class="fas fa-arrow-right"></i>
                </button>
                
                <p class="auth-switch">
                    Don't have an account? 
                    <a href="#" onclick="showAuthModal('signup'); return false;">Create one</a>
                </p>
            </form>
            
            <!-- Signup Form -->
            <form class="signup-form" style="display: none;" onsubmit="handleSignup(event)">
                <div class="auth-input-group">
                    <label for="signupName">Full Name</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-user"></i>
                        <input type="text" id="signupName" placeholder="Enter your name" required>
                    </div>
                </div>
                
                <div class="auth-input-group">
                    <label for="signupEmail">Email</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-envelope"></i>
                        <input type="email" id="signupEmail" placeholder="Enter your email" required>
                    </div>
                </div>
                
                <div class="auth-input-group">
                    <label for="signupPassword">Password</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="signupPassword" placeholder="Create a password (min 6 chars)" required minlength="6">
                        <button type="button" class="password-toggle" onclick="togglePassword('signupPassword')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                </div>
                
                <div class="auth-input-group">
                    <label for="signupConfirmPassword">Confirm Password</label>
                    <div class="auth-input-wrapper">
                        <i class="fas fa-lock"></i>
                        <input type="password" id="signupConfirmPassword" placeholder="Confirm your password" required>
                        <button type="button" class="password-toggle" onclick="togglePassword('signupConfirmPassword')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                </div>
                
                <button type="submit" class="auth-submit-btn">
                    <span>Create Account</span>
                    <i class="fas fa-user-plus"></i>
                </button>
                
                <p class="auth-switch">
                    Already have an account? 
                    <a href="#" onclick="showAuthModal('login'); return false;">Sign in</a>
                </p>
            </form>
        </div>
    `;
    
    return modal;
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    const icon = input.parentElement.querySelector('.password-toggle i');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        input.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const submitBtn = event.target.querySelector('.auth-submit-btn');
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing in...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentUser = data.user;
            showNotification(`Welcome back, ${currentUser.name}!`, 'success');
            closeAuthModal();
            updateUIForLoggedInUser();
        } else {
            showNotification(data.error || 'Login failed', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showNotification('Failed to connect to server', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Sign In</span><i class="fas fa-arrow-right"></i>';
    }
}

async function handleSignup(event) {
    event.preventDefault();
    
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;
    const submitBtn = event.target.querySelector('.auth-submit-btn');
    
    if (password !== confirmPassword) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({ name, email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentUser = data.user;
            showNotification(`Welcome, ${currentUser.name}! Account created successfully.`, 'success');
            closeAuthModal();
            updateUIForLoggedInUser();
        } else {
            showNotification(data.error || 'Signup failed', 'error');
        }
    } catch (error) {
        console.error('Signup error:', error);
        showNotification('Failed to connect to server', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Create Account</span><i class="fas fa-user-plus"></i>';
    }
}

async function handleLogout() {
    try {
        await fetch(`${API_BASE_URL}/api/auth/logout`, { 
            method: 'POST',
            credentials: 'include'
        });
        currentUser = null;
        updateUIForLoggedOutUser();
        closeUserMenu();
        showNotification('Logged out successfully', 'success');
    } catch (error) {
        console.error('Logout error:', error);
    }
}

function showUserMenu() {
    let menu = document.getElementById('userMenu');
    if (!menu) {
        menu = document.createElement('div');
        menu.id = 'userMenu';
        menu.className = 'user-menu';
        menu.innerHTML = `
            <div class="user-menu-header">
                <div class="user-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="user-info">
                    <span class="user-name">${currentUser?.name || 'User'}</span>
                    <span class="user-email">${currentUser?.email || ''}</span>
                </div>
            </div>
            <div class="user-menu-items">
                <button type="button" onclick="showHistoryPanel()">
                    <i class="fas fa-history"></i>
                    <span>Scan History</span>
                </button>
                <button type="button" onclick="handleLogout()">
                    <i class="fas fa-sign-out-alt"></i>
                    <span>Sign Out</span>
                </button>
            </div>
        `;
        document.body.appendChild(menu);
    }
    
    // Update user info
    menu.querySelector('.user-name').textContent = currentUser?.name || 'User';
    menu.querySelector('.user-email').textContent = currentUser?.email || '';
    
    // Position the menu
    const authButton = document.getElementById('authButton');
    if (authButton) {
        const rect = authButton.getBoundingClientRect();
        menu.style.top = `${rect.bottom + 10}px`;
        menu.style.right = `${window.innerWidth - rect.right}px`;
    }
    
    menu.classList.toggle('show');
    
    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', closeUserMenuOnOutsideClick);
    }, 100);
}

function closeUserMenu() {
    const menu = document.getElementById('userMenu');
    if (menu) {
        menu.classList.remove('show');
    }
    document.removeEventListener('click', closeUserMenuOnOutsideClick);
}

function closeUserMenuOnOutsideClick(event) {
    const menu = document.getElementById('userMenu');
    const authButton = document.getElementById('authButton');
    
    if (menu && !menu.contains(event.target) && !authButton?.contains(event.target)) {
        closeUserMenu();
    }
}

// ==================== Scan History System ====================
async function showHistoryPanel() {
    closeUserMenu();
    
    let panel = document.getElementById('historyPanel');
    if (!panel) {
        panel = createHistoryPanel();
        document.body.appendChild(panel);
    }
    
    panel.classList.add('show');
    document.body.style.overflow = 'hidden';
    
    await loadScanHistory();
}

function createHistoryPanel() {
    const panel = document.createElement('div');
    panel.id = 'historyPanel';
    panel.className = 'history-panel';
    panel.innerHTML = `
        <div class="history-panel-overlay" onclick="closeHistoryPanel()"></div>
        <div class="history-panel-content">
            <div class="history-panel-header">
                <h2><i class="fas fa-history"></i> Scan History</h2>
                <button class="history-close-btn" type="button" onclick="closeHistoryPanel()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="history-panel-body" id="historyList">
                <div class="history-loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Loading history...</span>
                </div>
            </div>
        </div>
    `;
    return panel;
}

function closeHistoryPanel() {
    const panel = document.getElementById('historyPanel');
    if (panel) {
        panel.classList.remove('show');
        document.body.style.overflow = '';
    }
}

async function loadScanHistory() {
    const historyList = document.getElementById('historyList');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/history`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok && data.scans && data.scans.length > 0) {
            historyList.innerHTML = data.scans.map(scan => `
                <div class="history-item" data-scan-id="${scan.id}">
                    <div class="history-item-image">
                        ${scan.original_image ? 
                            `<img src="${scan.original_image}" alt="Scan thumbnail">` :
                            `<div class="no-image"><i class="fas fa-brain"></i></div>`
                        }
                    </div>
                    <div class="history-item-info">
                        <div class="history-item-result ${scan.has_tumor ? 'tumor-detected' : 'no-tumor'}">
                            ${scan.has_tumor ? (scan.severity || 'Tumor Detected') : 'No Tumor'}
                        </div>
                        <div class="history-item-confidence">
                            Confidence: ${(scan.confidence * 100).toFixed(1)}%
                        </div>
                        <div class="history-item-date">
                            <i class="fas fa-calendar"></i>
                            ${formatDate(scan.date)}
                        </div>
                    </div>
                    <div class="history-item-actions">
                        <button class="history-view-btn" type="button" onclick="viewHistoryScan('${scan.id}')" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="history-delete-btn" type="button" onclick="deleteHistoryScan('${scan.id}')" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `).join('');
        } else if (response.status === 401) {
            historyList.innerHTML = `
                <div class="history-empty">
                    <i class="fas fa-sign-in-alt"></i>
                    <h3>Login Required</h3>
                    <p>Please login to view your scan history.</p>
                    <button type="button" onclick="closeHistoryPanel(); showAuthModal('login');" class="auth-btn">Sign In</button>
                </div>
            `;
        } else {
            historyList.innerHTML = `
                <div class="history-empty">
                    <i class="fas fa-folder-open"></i>
                    <h3>No Scans Yet</h3>
                    <p>Your scan history will appear here after you save your first analysis.</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load history:', error);
        historyList.innerHTML = `
            <div class="history-error">
                <i class="fas fa-exclamation-circle"></i>
                <h3>Failed to Load History</h3>
                <p>Please try again later.</p>
            </div>
        `;
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    // Less than 24 hours
    if (diff < 86400000) {
        const hours = Math.floor(diff / 3600000);
        if (hours < 1) {
            const minutes = Math.floor(diff / 60000);
            return minutes < 1 ? 'Just now' : `${minutes} min ago`;
        }
        return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    }
    
    // Less than 7 days
    if (diff < 604800000) {
        const days = Math.floor(diff / 86400000);
        return `${days} day${days > 1 ? 's' : ''} ago`;
    }
    
    // Otherwise show full date
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

async function viewHistoryScan(scanId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/history/${scanId}`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok && data.scan) {
            closeHistoryPanel();
            
            // Display the historical results - the scan data is directly in data.scan
            currentResults = data.scan;
            displayResults(currentResults);
            
            // Scroll to results section
            const resultsSection = document.getElementById('resultsContainer') || document.querySelector('#scan');
            if (resultsSection) {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            showNotification('Failed to load scan details', 'error');
        }
    } catch (error) {
        console.error('Error loading scan:', error);
        showNotification('Failed to load scan details', 'error');
    }
}

async function deleteHistoryScan(scanId) {
    if (!confirm('Are you sure you want to delete this scan from history?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/history/${scanId}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        if (response.ok) {
            showNotification('Scan deleted from history', 'success');
            await loadScanHistory();
        } else {
            showNotification('Failed to delete scan', 'error');
        }
    } catch (error) {
        console.error('Error deleting scan:', error);
        showNotification('Failed to delete scan', 'error');
    }
}

// Check auth status on page load
document.addEventListener('DOMContentLoaded', () => {
    checkAuthStatus();
});

// ==================== Export functions to global scope ====================
// These need to be global for onclick handlers in dynamically generated HTML
window.shareReport = shareReport;
window.downloadReportAsPDF = downloadReportAsPDF;
window.printReport = printReport;
window.saveToHistory = saveToHistory;
window.showAuthModal = showAuthModal;
window.closeAuthModal = closeAuthModal;
window.handleLogin = handleLogin;
window.handleSignup = handleSignup;
window.handleLogout = handleLogout;
window.showHistoryPanel = showHistoryPanel;
window.closeHistoryPanel = closeHistoryPanel;
window.viewHistoryScan = viewHistoryScan;
window.deleteHistoryScan = deleteHistoryScan;
window.togglePassword = togglePassword;

// ==================== Export for debugging ====================
window.NeuroScanAI = {
    checkHealth: checkAPIHealth,
    currentResults,
    uploadedFile,
    currentUser,
    showAuthModal,
    showHistoryPanel,
    saveToHistory
};
