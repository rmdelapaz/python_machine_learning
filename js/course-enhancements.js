/**
 * Course Enhancements for Machine Learning Course
 * Adds various UI improvements and features
 */

(function() {
    'use strict';
    
    document.addEventListener('DOMContentLoaded', function() {
        
        // Add reading time estimation
        function addReadingTime() {
            const content = document.querySelector('main');
            if (!content) return;
            
            const text = content.textContent || content.innerText;
            const wordsPerMinute = 200;
            const words = text.split(/\s+/).length;
            const readingTime = Math.ceil(words / wordsPerMinute);
            
            const readingTimeElement = document.querySelector('.reading-time');
            if (readingTimeElement) {
                readingTimeElement.textContent = `Estimated reading time: ${readingTime} minutes`;
            }
        }
        
        // Add progress indicator
        function updateProgressBar() {
            const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            
            const progressBar = document.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = scrolled + '%';
            }
        }
        
        // Add smooth scrolling for anchor links
        function addSmoothScrolling() {
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
        }
        
        // Add keyboard navigation
        function addKeyboardNavigation() {
            document.addEventListener('keydown', function(e) {
                // Alt + Left: Previous lesson
                if (e.altKey && e.key === 'ArrowLeft') {
                    const prevLink = document.querySelector('.prev-lesson');
                    if (prevLink) prevLink.click();
                }
                
                // Alt + Right: Next lesson
                if (e.altKey && e.key === 'ArrowRight') {
                    const nextLink = document.querySelector('.next-lesson');
                    if (nextLink) nextLink.click();
                }
                
                // Alt + Home: Course home
                if (e.altKey && e.key === 'Home') {
                    const homeLink = document.querySelector('.home-link');
                    if (homeLink) homeLink.click();
                }
            });
        }
        
        // Initialize features
        addReadingTime();
        addSmoothScrolling();
        addKeyboardNavigation();
        
        // Update progress bar on scroll
        window.addEventListener('scroll', updateProgressBar);
        
        // Initial progress bar update
        updateProgressBar();
        
    });
    
})();