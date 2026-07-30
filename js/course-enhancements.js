/**
 * Course Enhancement JavaScript
 * Provides interactive features, smooth scrolling, progress tracking, and more
 */

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Course enhancements loaded');
    
    // Initialize all enhancements
    initProgressIndicator();
    initSmoothScrolling();
    initCodeCopyButtons();
    initInteractiveTOC();
    initSearchFunctionality();
    initKeyboardShortcuts();
    initThemeToggle();
    initLessonProgress();
    initQuizInteractivity();
    initMobileMenu();
    initAccessibilityFeatures();
    initPrintStyles();
    initAnalytics();
});

/**
 * Progress Indicator - Shows reading progress
 */
function initProgressIndicator() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;
    
    window.addEventListener('scroll', () => {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight - windowHeight;
        const scrolled = window.scrollY;
        const progress = (scrolled / documentHeight) * 100;
        
        progressBar.style.width = `${Math.min(progress, 100)}%`;
        
        // Update ARIA attributes for accessibility
        const progressIndicator = document.querySelector('.progress-indicator');
        if (progressIndicator) {
            progressIndicator.setAttribute('aria-valuenow', Math.round(progress));
        }
    });
}

/**
 * Smooth Scrolling for internal links
 */
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerOffset = 70;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
                
                // Update URL without jumping
                history.pushState(null, null, this.getAttribute('href'));
            }
        });
    });
}

/**
 * Code Copy Buttons - Already handled by clipboard.js
 * This function enhances it with notifications
 */
function initCodeCopyButtons() {
    // Add visual feedback for copy action
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', function() {
            const originalText = this.textContent;
            this.textContent = 'Copied!';
            this.classList.add('copied');
            
            setTimeout(() => {
                this.textContent = originalText;
                this.classList.remove('copied');
            }, 2000);
        });
    });
}

/**
 * Interactive Table of Contents
 */
function initInteractiveTOC() {
    // Highlight current section in TOC
    const tocLinks = document.querySelectorAll('.toc-link');
    const sections = document.querySelectorAll('h2, h3');
    
    if (tocLinks.length === 0 || sections.length === 0) return;
    
    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    };
    
    const observerCallback = (entries) => {
        entries.forEach(entry => {
            const id = entry.target.getAttribute('id');
            const tocLink = document.querySelector(`.toc-link[href="#${id}"]`);
            
            if (tocLink) {
                if (entry.isIntersecting) {
                    // Remove active class from all links
                    tocLinks.forEach(link => link.classList.remove('active'));
                    // Add active class to current link
                    tocLink.classList.add('active');
                }
            }
        });
    };
    
    const observer = new IntersectionObserver(observerCallback, observerOptions);
    sections.forEach(section => observer.observe(section));
}

/**
 * Search Functionality
 */
function initSearchFunctionality() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    let searchIndex = [];
    
    // Build search index from page content
    function buildSearchIndex() {
        const content = document.querySelectorAll('h1, h2, h3, p, li');
        content.forEach((element, index) => {
            searchIndex.push({
                id: index,
                text: element.textContent.toLowerCase(),
                element: element,
                type: element.tagName.toLowerCase()
            });
        });
    }
    
    // Perform search
    function performSearch(query) {
        const results = [];
        const searchQuery = query.toLowerCase().trim();
        
        if (searchQuery.length < 2) {
            searchResults.innerHTML = '';
            searchResults.style.display = 'none';
            return;
        }
        
        searchIndex.forEach(item => {
            if (item.text.includes(searchQuery)) {
                results.push(item);
            }
        });
        
        displayResults(results.slice(0, 10)); // Show top 10 results
    }
    
    // Display search results
    function displayResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No results found</div>';
            searchResults.style.display = 'block';
            return;
        }
        
        let html = '<ul class="search-results-list">';
        results.forEach(result => {
            const excerpt = getExcerpt(result.text, searchInput.value.toLowerCase());
            html += `
                <li class="search-result-item" data-index="${result.id}">
                    <span class="result-type">${result.type.toUpperCase()}</span>
                    <span class="result-text">${excerpt}</span>
                </li>
            `;
        });
        html += '</ul>';
        
        searchResults.innerHTML = html;
        searchResults.style.display = 'block';
        
        // Add click handlers to results
        document.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', function() {
                const index = this.getAttribute('data-index');
                const element = searchIndex[index].element;
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.classList.add('highlight');
                setTimeout(() => element.classList.remove('highlight'), 2000);
                
                // Clear search
                searchInput.value = '';
                searchResults.style.display = 'none';
            });
        });
    }
    
    // Get excerpt with highlighted search term
    function getExcerpt(text, query) {
        const index = text.indexOf(query);
        const start = Math.max(0, index - 30);
        const end = Math.min(text.length, index + query.length + 30);
        
        let excerpt = text.substring(start, end);
        if (start > 0) excerpt = '...' + excerpt;
        if (end < text.length) excerpt = excerpt + '...';
        
        // Highlight the search term
        const regex = new RegExp(query, 'gi');
        excerpt = excerpt.replace(regex, `<mark>$&</mark>`);
        
        return excerpt;
    }
    
    // Initialize
    buildSearchIndex();
    
    // Event listeners
    searchInput.addEventListener('input', (e) => {
        performSearch(e.target.value);
    });
    
    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });
}

/**
 * Keyboard Shortcuts
 */
function initKeyboardShortcuts() {
    const shortcuts = {
        '/': () => {
            // Focus search
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.focus();
                return false; // Prevent default
            }
        },
        'Escape': () => {
            // Close search or modal
            const searchResults = document.getElementById('search-results');
            if (searchResults) {
                searchResults.style.display = 'none';
            }
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.blur();
            }
        },
        'ArrowLeft': () => {
            // Previous lesson
            const prevLink = document.querySelector('.prev-lesson');
            if (prevLink) {
                window.location.href = prevLink.href;
            }
        },
        'ArrowRight': () => {
            // Next lesson
            const nextLink = document.querySelector('.next-lesson');
            if (nextLink) {
                window.location.href = nextLink.href;
            }
        },
        'h': () => {
            // Go home
            window.location.href = '/index.html';
        },
        '?': () => {
            // Show keyboard shortcuts help
            showKeyboardShortcuts();
        }
    };
    
    document.addEventListener('keydown', (e) => {
        // Don't trigger shortcuts when typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        const key = e.key;
        if (shortcuts[key]) {
            const preventDefault = shortcuts[key]();
            if (preventDefault === false) {
                e.preventDefault();
            }
        }
    });
    
    function showKeyboardShortcuts() {
        const modal = document.createElement('div');
        modal.className = 'shortcuts-modal';
        modal.innerHTML = `
            <div class="shortcuts-content">
                <h3>Keyboard Shortcuts</h3>
                <button class="close-modal">&times;</button>
                <dl>
                    <dt>/</dt><dd>Focus search</dd>
                    <dt>ESC</dt><dd>Close search/modal</dd>
                    <dt>←</dt><dd>Previous lesson</dd>
                    <dt>→</dt><dd>Next lesson</dd>
                    <dt>H</dt><dd>Go to home</dd>
                    <dt>?</dt><dd>Show this help</dd>
                </dl>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal
        modal.querySelector('.close-modal').addEventListener('click', () => {
            modal.remove();
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }
}

/**
 * Theme Toggle (Light/Dark mode)
 */
function initThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    
    // Check for saved theme preference or default to light
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update button text/icon
        themeToggle.textContent = newTheme === 'light' ? '🌙' : '☀️';
        
        // Smooth transition
        document.documentElement.style.transition = 'background-color 0.3s, color 0.3s';
    });
}

/**
 * Lesson Progress Tracking
 */
function initLessonProgress() {
    const currentPage = window.location.pathname.split('/').pop().replace('.html', '');
    if (!currentPage) return;
    
    // Get or initialize progress data
    let progress = JSON.parse(localStorage.getItem('lessonProgress') || '{}');
    
    // Mark current page as visited
    progress[currentPage] = {
        visited: true,
        lastVisited: new Date().toISOString(),
        scrollPosition: 0
    };
    
    // Save progress
    localStorage.setItem('lessonProgress', JSON.stringify(progress));
    
    
    // Save scroll position on page leave
    window.addEventListener('beforeunload', () => {
        progress[currentPage].scrollPosition = window.scrollY;
        localStorage.setItem('lessonProgress', JSON.stringify(progress));
    });
    
    // Restore scroll position
    const savedPosition = progress[currentPage]?.scrollPosition;
    if (savedPosition && savedPosition > 0) {
        // Wait for content to load
        setTimeout(() => {
            window.scrollTo(0, savedPosition);
        }, 100);
    }
}


/**
 * Quiz Interactivity
 */
function initQuizInteractivity() {
    const quizzes = document.querySelectorAll('.quiz-container');
    
    quizzes.forEach(quiz => {
        const questions = quiz.querySelectorAll('.quiz-question');
        
        questions.forEach(question => {
            const options = question.querySelectorAll('.quiz-option');
            const feedback = question.querySelector('.quiz-feedback');
            
            options.forEach(option => {
                option.addEventListener('click', () => {
                    // Remove previous selections
                    options.forEach(opt => opt.classList.remove('selected', 'correct', 'incorrect'));
                    
                    // Mark selected
                    option.classList.add('selected');
                    
                    // Check answer
                    const isCorrect = option.dataset.correct === 'true';
                    if (isCorrect) {
                        option.classList.add('correct');
                        if (feedback) {
                            feedback.textContent = 'Correct! ' + (option.dataset.explanation || '');
                            feedback.className = 'quiz-feedback correct';
                        }
                    } else {
                        option.classList.add('incorrect');
                        if (feedback) {
                            feedback.textContent = 'Try again. ' + (option.dataset.hint || '');
                            feedback.className = 'quiz-feedback incorrect';
                        }
                    }
                });
            });
        });
    });
}

/**
 * Mobile Menu Toggle
 */
function initMobileMenu() {
    const menuToggle = document.getElementById('mobile-menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (!menuToggle || !navLinks) return;
    
    menuToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        menuToggle.classList.toggle('active');
        
        // Update ARIA attributes
        const isOpen = navLinks.classList.contains('active');
        menuToggle.setAttribute('aria-expanded', isOpen);
        navLinks.setAttribute('aria-hidden', !isOpen);
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!menuToggle.contains(e.target) && !navLinks.contains(e.target)) {
            navLinks.classList.remove('active');
            menuToggle.classList.remove('active');
        }
    });
}

/**
 * Accessibility Features
 */
function initAccessibilityFeatures() {
    // Skip to main content
    const skipLink = document.querySelector('.skip-to-main');
    if (skipLink) {
        skipLink.addEventListener('click', (e) => {
            e.preventDefault();
            const main = document.getElementById('main-content');
            if (main) {
                main.tabIndex = -1;
                main.focus();
            }
        });
    }
    
    // Announce page changes for screen readers
    const announcer = document.createElement('div');
    announcer.className = 'sr-only';
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    document.body.appendChild(announcer);
    
    // Improve focus visibility
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }
    });
    
    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-nav');
    });
    
    // Add ARIA labels to interactive elements
    document.querySelectorAll('button, a, input').forEach(element => {
        if (!element.getAttribute('aria-label') && !element.textContent.trim()) {
            // Try to find a meaningful label
            const title = element.getAttribute('title');
            if (title) {
                element.setAttribute('aria-label', title);
            }
        }
    });
}

/**
 * Print Styles Enhancement
 */
function initPrintStyles() {
    // Add print button
    const printButton = document.getElementById('print-button');
    if (printButton) {
        printButton.addEventListener('click', () => {
            window.print();
        });
    }
    
    // Prepare content for printing
    window.addEventListener('beforeprint', () => {
        // Expand all collapsible sections
        document.querySelectorAll('details').forEach(details => {
            details.setAttribute('open', 'true');
        });
        
        // Remove interactive elements
        document.body.classList.add('printing');
    });
    
    window.addEventListener('afterprint', () => {
        document.body.classList.remove('printing');
    });
}

/**
 * Simple Analytics (Privacy-friendly)
 */
function initAnalytics() {
    // Track page views
    const pageView = {
        page: window.location.pathname,
        timestamp: new Date().toISOString(),
        referrer: document.referrer
    };
    
    // Store locally (no external tracking)
    let analytics = JSON.parse(localStorage.getItem('courseAnalytics') || '[]');
    analytics.push(pageView);
    
    // Keep only last 100 entries
    if (analytics.length > 100) {
        analytics = analytics.slice(-100);
    }
    
    localStorage.setItem('courseAnalytics', JSON.stringify(analytics));
    
    // Track time on page
    const startTime = Date.now();
    window.addEventListener('beforeunload', () => {
        const timeSpent = Math.round((Date.now() - startTime) / 1000);
        console.log(`Time on page: ${timeSpent} seconds`);
    });
}

/**
 * Utility Functions
 */

// Debounce function for performance
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

// Throttle function for performance
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export for use in other scripts
window.courseEnhancements = {
    debounce,
    throttle,
    initProgressIndicator,
    initSmoothScrolling
};

console.log('Course enhancements initialized successfully');
