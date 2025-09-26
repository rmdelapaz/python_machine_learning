/**
 * Copy to Clipboard functionality for code blocks
 * Adds a copy button to all code blocks and handles the copy action
 */

(function() {
    'use strict';
    
    // Wait for DOM to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        
        // Find all code blocks
        const codeBlocks = document.querySelectorAll('pre code');
        
        codeBlocks.forEach(function(codeBlock) {
            // Create wrapper div for positioning
            const wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            wrapper.style.position = 'relative';
            
            // Wrap the pre element
            const preElement = codeBlock.parentElement;
            preElement.parentNode.insertBefore(wrapper, preElement);
            wrapper.appendChild(preElement);
            
            // Create copy button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = 'Copy';
            copyButton.setAttribute('aria-label', 'Copy code to clipboard');
            
            // Style the button
            copyButton.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                padding: 6px 12px;
                background-color: #4a5568;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                cursor: pointer;
                opacity: 0.8;
                transition: opacity 0.2s, background-color 0.2s;
                z-index: 10;
            `;
            
            // Add hover effect
            copyButton.addEventListener('mouseenter', function() {
                this.style.opacity = '1';
                this.style.backgroundColor = '#2d3748';
            });
            
            copyButton.addEventListener('mouseleave', function() {
                this.style.opacity = '0.8';
                this.style.backgroundColor = '#4a5568';
            });
            
            // Add copy functionality
            copyButton.addEventListener('click', function() {
                const textToCopy = codeBlock.textContent || codeBlock.innerText;
                
                // Modern clipboard API
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(textToCopy).then(function() {
                        // Success feedback
                        showCopyFeedback(copyButton, true);
                    }).catch(function(err) {
                        // Fallback to older method
                        fallbackCopyTextToClipboard(textToCopy, copyButton);
                    });
                } else {
                    // Fallback for older browsers
                    fallbackCopyTextToClipboard(textToCopy, copyButton);
                }
            });
            
            // Add button to wrapper
            wrapper.appendChild(copyButton);
        });
    });
    
    // Fallback copy method for older browsers
    function fallbackCopyTextToClipboard(text, button) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        
        // Avoid scrolling to bottom
        textArea.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 2em;
            height: 2em;
            padding: 0;
            border: none;
            outline: none;
            box-shadow: none;
            background: transparent;
        `;
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            const successful = document.execCommand('copy');
            showCopyFeedback(button, successful);
        } catch (err) {
            showCopyFeedback(button, false);
        }
        
        document.body.removeChild(textArea);
    }
    
    // Show feedback when copy is complete
    function showCopyFeedback(button, success) {
        const originalText = button.textContent;
        
        if (success) {
            button.textContent = '✓ Copied!';
            button.style.backgroundColor = '#48bb78';
        } else {
            button.textContent = '✗ Failed';
            button.style.backgroundColor = '#f56565';
        }
        
        // Reset button after 2 seconds
        setTimeout(function() {
            button.textContent = originalText;
            button.style.backgroundColor = '#4a5568';
        }, 2000);
    }
    
})();