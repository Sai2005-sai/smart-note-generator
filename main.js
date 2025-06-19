// Smart Note Generator - Main JavaScript File

document.addEventListener('DOMContentLoaded', function() {
    console.log('Smart Note Generator initialized');
    
    // Initialize page animations
    initializeAnimations();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize copy functionality
    initializeCopyFunctionality();
    
    // Initialize responsive features
    initializeResponsiveFeatures();
});

/**
 * Initialize page animations
 */
function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Animate feature icons on scroll
    if (typeof IntersectionObserver !== 'undefined') {
        const observerOptions = {
            threshold: 0.5,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'pulse 1s ease-in-out';
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('.feature-icon').forEach(icon => {
            observer.observe(icon);
        });
    }
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const form = document.getElementById('videoForm');
    const urlInput = document.getElementById('youtube_url');
    
    if (!form || !urlInput) return;
    
    // Disabled real-time validation for better compatibility
    // urlInput.addEventListener('input', function() {
    //     validateYouTubeUrl(this.value);
    // });
    
    // Disabled form validation to prevent interference
    // form.addEventListener('submit', function(e) {
    //     const url = urlInput.value.trim();
    //     
    //     if (!url) {
    //         e.preventDefault();
    //         showError('Please enter a YouTube URL');
    //         return false;
    //     }
    //     
    //     // Show loading state
    //     showLoadingState(this);
    //     return true;
    // });
}

/**
 * Validate YouTube URL
 */
function validateYouTubeUrl(url) {
    if (!url) return false;
    
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/;
    const isValid = youtubeRegex.test(url);
    
    const input = document.getElementById('youtube_url');
    const feedbackDiv = input.parentNode.parentNode.querySelector('.invalid-feedback') || 
                       document.createElement('div');
    
    if (!feedbackDiv.classList.contains('invalid-feedback')) {
        feedbackDiv.className = 'invalid-feedback';
        input.parentNode.parentNode.appendChild(feedbackDiv);
    }
    
    if (url && !isValid) {
        input.classList.add('is-invalid');
        feedbackDiv.textContent = 'Please enter a valid YouTube URL';
        feedbackDiv.style.display = 'block';
        return false;
    } else {
        input.classList.remove('is-invalid');
        feedbackDiv.style.display = 'none';
        return true;
    }
}

/**
 * Show loading state
 */
function showLoadingState(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    const spinner = submitBtn.querySelector('.spinner-border');
    const icon = submitBtn.querySelector('i');
    
    // Disable form
    submitBtn.disabled = true;
    form.querySelectorAll('input').forEach(input => input.disabled = true);
    
    // Update button
    if (spinner) spinner.classList.remove('d-none');
    if (icon) icon.className = 'fas fa-hourglass-half me-2';
    
    // Show loading overlay
    showLoadingOverlay();
}

/**
 * Show loading overlay with progress simulation
 */
function showLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) return;
    
    const progressBar = overlay.querySelector('.progress-bar');
    const loadingText = document.getElementById('loadingText');
    
    overlay.style.display = 'flex';
    
    // Progress simulation
    let progress = 0;
    const steps = [
        { progress: 20, text: 'Analyzing video URL...' },
        { progress: 40, text: 'Extracting content...' },
        { progress: 60, text: 'Processing with AI...' },
        { progress: 80, text: 'Generating summary...' },
        { progress: 95, text: 'Finalizing notes...' }
    ];
    
    let currentStep = 0;
    
    const updateProgress = () => {
        if (currentStep < steps.length) {
            const step = steps[currentStep];
            progress = step.progress;
            
            if (progressBar) {
                progressBar.style.width = progress + '%';
            }
            
            if (loadingText) {
                loadingText.textContent = step.text;
            }
            
            currentStep++;
            setTimeout(updateProgress, 1500 + Math.random() * 1000);
        }
    };
    
    updateProgress();
}

/**
 * Initialize copy functionality
 */
function initializeCopyFunctionality() {
    // Add copy buttons to code blocks or text areas if needed
    const copyButtons = document.querySelectorAll('[data-copy]');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-copy');
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                copyToClipboard(targetElement.textContent);
                showCopySuccess(this);
            }
        });
    });
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        return navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return Promise.resolve();
    }
}

/**
 * Show copy success feedback
 */
function showCopySuccess(button) {
    const originalContent = button.innerHTML;
    const originalClass = button.className;
    
    button.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
    button.className = button.className.replace('btn-outline-primary', 'btn-success');
    
    setTimeout(() => {
        button.innerHTML = originalContent;
        button.className = originalClass;
    }, 2000);
}

/**
 * Initialize responsive features
 */
function initializeResponsiveFeatures() {
    // Handle mobile navigation if needed
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            // Scrolling down
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // Scrolling up
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScrollTop = scrollTop;
    });
    
    // Add smooth transitions
    navbar.style.transition = 'transform 0.3s ease';
    
    // Handle window resize
    window.addEventListener('resize', handleResize);
    handleResize(); // Initial call
}

/**
 * Handle window resize
 */
function handleResize() {
    const width = window.innerWidth;
    
    // Adjust hero text size on mobile
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        if (width < 576) {
            heroTitle.style.fontSize = '2rem';
        } else if (width < 768) {
            heroTitle.style.fontSize = '2.5rem';
        } else {
            heroTitle.style.fontSize = '3.5rem';
        }
    }
    
    // Adjust card padding on mobile
    const mainCard = document.querySelector('.main-card .card-body');
    if (mainCard) {
        if (width < 768) {
            mainCard.style.padding = '2rem';
        } else {
            mainCard.style.padding = '3rem';
        }
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Create or update error alert
    let errorAlert = document.querySelector('.alert-danger');
    
    if (!errorAlert) {
        errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger alert-dismissible fade show';
        errorAlert.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            <span class="alert-message">${message}</span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(errorAlert, container.firstChild);
        }
    } else {
        errorAlert.querySelector('.alert-message').textContent = message;
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (errorAlert && errorAlert.parentNode) {
            errorAlert.remove();
        }
    }, 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    const successAlert = document.createElement('div');
    successAlert.className = 'alert alert-success alert-dismissible fade show';
    successAlert.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(successAlert, container.firstChild);
    }
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        if (successAlert && successAlert.parentNode) {
            successAlert.remove();
        }
    }, 3000);
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Smooth scroll to element
 */
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Export functions for global use
window.SmartNoteGenerator = {
    validateYouTubeUrl,
    showError,
    showSuccess,
    copyToClipboard,
    smoothScrollTo
};
