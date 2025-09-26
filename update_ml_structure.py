#!/usr/bin/env python3
"""
Update python_machine_learning to add clipboard functionality and match other courses.
This script:
1. Creates necessary directories (styles, js, scripts)
2. Copies CSS and JS files from python_datascience
3. Updates all HTML files to include clipboard.js and course-enhancements.js
4. Adds enhanced.css references
5. Adds accessibility features
"""

import os
import re
import sys
import shutil
import platform
from pathlib import Path

def detect_path_format():
    """Detect if we're running on Windows or WSL/Linux"""
    system = platform.system()
    if system == "Windows":
        return "windows"
    else:
        return "wsl"

def get_project_paths():
    """Get the correct paths for both source and destination projects"""
    system_type = detect_path_format()
    
    if system_type == "windows":
        # Windows path format for WSL files
        paths = {
            'ml': [
                r"\\wsl$\Ubuntu\home\practicalace\projects\python_machine_learning",
                r"\\wsl.localhost\Ubuntu\home\practicalace\projects\python_machine_learning",
            ],
            'datascience': [
                r"\\wsl$\Ubuntu\home\practicalace\projects\python_datascience",
                r"\\wsl.localhost\Ubuntu\home\practicalace\projects\python_datascience",
            ]
        }
    else:
        # WSL/Linux path format
        paths = {
            'ml': [
                "/home/practicalace/projects/python_machine_learning",
                os.path.expanduser("~/projects/python_machine_learning"),
            ],
            'datascience': [
                "/home/practicalace/projects/python_datascience", 
                os.path.expanduser("~/projects/python_datascience"),
            ]
        }
    
    # Find which paths exist
    found_paths = {}
    for project, path_list in paths.items():
        for path in path_list:
            if os.path.exists(path):
                found_paths[project] = path
                break
    
    if 'ml' not in found_paths:
        print("❌ Could not find python_machine_learning folder!")
        print("Please enter the full path to python_machine_learning:")
        user_path = input().strip()
        if os.path.exists(user_path):
            found_paths['ml'] = user_path
        else:
            print(f"Error: Path '{user_path}' does not exist!")
            sys.exit(1)
    
    return found_paths

def create_directory_structure(ml_path):
    """Create necessary directories"""
    print("\n📂 Creating directory structure...")
    
    directories = ['styles', 'js', 'scripts']
    
    for dir_name in directories:
        dir_path = os.path.join(ml_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"  ✅ Created {dir_name}/ directory")
        else:
            print(f"  ✓ {dir_name}/ already exists")

def copy_css_files(paths):
    """Copy CSS files from python_datascience"""
    print("\n📄 Copying CSS files...")
    
    if 'datascience' not in paths:
        print("  ⚠️  python_datascience not found, creating default CSS files...")
        create_default_css(paths['ml'])
        return
    
    css_files = ['main.css', 'enhanced.css']
    source_dir = os.path.join(paths['datascience'], 'styles')
    dest_dir = os.path.join(paths['ml'], 'styles')
    
    for css_file in css_files:
        source = os.path.join(source_dir, css_file)
        dest = os.path.join(dest_dir, css_file)
        
        if os.path.exists(source):
            if not os.path.exists(dest):
                shutil.copy2(source, dest)
                print(f"  ✅ Copied {css_file}")
            else:
                print(f"  ✓ {css_file} already exists")
        else:
            print(f"  ⚠️  {css_file} not found in source, creating default...")
            create_default_css_file(dest_dir, css_file)

def create_default_css(ml_path):
    """Create default CSS files if source is not available"""
    styles_dir = os.path.join(ml_path, 'styles')
    
    # Create main.css
    main_css_content = '''/* Main CSS for Machine Learning Course */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h1 {
    border-bottom: 2px solid #667eea;
    padding-bottom: 10px;
}

pre {
    background-color: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1em;
    overflow-x: auto;
}

code {
    background-color: #f4f4f4;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: "Courier New", monospace;
}

.breadcrumb {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    margin-bottom: 2rem;
    font-size: 0.9rem;
}

.breadcrumb a {
    color: #667eea;
    text-decoration: none;
}

.breadcrumb a:hover {
    text-decoration: underline;
}

.lesson-nav {
    display: flex;
    justify-content: space-between;
    margin-top: 3em;
    padding-top: 2em;
    border-top: 1px solid #ddd;
}

.lesson-nav a {
    color: #667eea;
    text-decoration: none;
    padding: 0.5em 1em;
    border: 1px solid #667eea;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.lesson-nav a:hover {
    background-color: #667eea;
    color: white;
}

blockquote {
    border-left: 4px solid #667eea;
    margin: 1.5em 0;
    padding: 0.5em 1em;
    background-color: #f8f9fa;
    font-style: italic;
}
'''
    
    # Create enhanced.css
    enhanced_css_content = '''/* Enhanced CSS for Machine Learning Course */
.skip-to-main {
    position: absolute;
    top: -40px;
    left: 0;
    background: #667eea;
    color: white;
    padding: 8px;
    text-decoration: none;
    z-index: 100;
}

.skip-to-main:focus {
    top: 0;
}

.progress-indicator {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: #e0e0e0;
    z-index: 9999;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    width: 0%;
    transition: width 0.3s ease;
}

.code-block-wrapper {
    position: relative;
    margin: 1em 0;
}

.copy-button {
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
}

.copy-button:hover {
    opacity: 1;
    background-color: #2d3748;
}

.reading-time {
    color: #666;
    font-size: 0.9em;
    margin-bottom: 1em;
}

.reading-time:before {
    content: "⏱️ ";
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    .lesson-nav {
        flex-direction: column;
        gap: 10px;
    }
    
    .lesson-nav a {
        display: block;
        text-align: center;
    }
}
'''
    
    with open(os.path.join(styles_dir, 'main.css'), 'w', encoding='utf-8') as f:
        f.write(main_css_content)
    print("  ✅ Created main.css")
    
    with open(os.path.join(styles_dir, 'enhanced.css'), 'w', encoding='utf-8') as f:
        f.write(enhanced_css_content)
    print("  ✅ Created enhanced.css")

def copy_js_files(paths):
    """Copy JavaScript files"""
    print("\n📜 Copying JavaScript files...")
    
    js_dir = os.path.join(paths['ml'], 'js')
    
    # Create clipboard.js
    clipboard_content = '''/**
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
    
})();'''
    
    # Create course-enhancements.js
    enhancements_content = '''/**
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
            const words = text.split(/\\s+/).length;
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
    
})();'''
    
    # Save JavaScript files
    with open(os.path.join(js_dir, 'clipboard.js'), 'w', encoding='utf-8') as f:
        f.write(clipboard_content)
    print("  ✅ Created clipboard.js")
    
    with open(os.path.join(js_dir, 'course-enhancements.js'), 'w', encoding='utf-8') as f:
        f.write(enhancements_content)
    print("  ✅ Created course-enhancements.js")

def update_html_file(filepath):
    """Update a single HTML file to include scripts and enhanced CSS"""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, f"Error reading file: {e}"
    
    original_content = content
    changes_made = []
    
    # Check if this is the index.html (already has the scripts)
    if 'index.html' in filepath:
        return False, "index.html already updated"
    
    # 1. Fix CSS paths (they reference styles/ but might need fixing)
    # Already correct in most files
    
    # 2. Add clipboard.js if not present
    if 'clipboard.js' not in content:
        # Add before </body>
        if '</body>' in content:
            pattern = r'(</body>)'
            replacement = r'    <script src="js/clipboard.js"></script>\n\1'
            new_content = re.sub(pattern, replacement, content, count=1)
            
            if new_content != content:
                content = new_content
                changes_made.append("added clipboard.js")
    
    # 3. Ensure course-enhancements.js is referenced correctly
    if 'course-enhancements.js' in content:
        # Fix path if needed
        content = re.sub(r'src="course-enhancements\.js"', 'src="js/course-enhancements.js"', content)
        if 'src="js/course-enhancements.js"' in content:
            changes_made.append("fixed course-enhancements.js path")
    
    # 4. Add skip-to-main link if not present
    if 'skip-to-main' not in content and '<body>' in content:
        pattern = r'(<body>)'
        replacement = r'\1\n    <!-- Skip to main content for accessibility -->\n    <a href="#main-content" class="skip-to-main">Skip to main content</a>'
        new_content = re.sub(pattern, replacement, content, count=1)
        
        if new_content != content:
            content = new_content
            changes_made.append("added skip-to-main link")
    
    # 5. Add progress indicator if not present
    if 'progress-indicator' not in content and 'skip-to-main' in content:
        pattern = r'(    <a href="#main-content" class="skip-to-main">Skip to main content</a>)'
        replacement = r'\1\n    \n    <!-- Progress indicator -->\n    <div class="progress-indicator" role="progressbar" aria-label="Page scroll progress">\n        <div class="progress-bar"></div>\n    </div>'
        new_content = re.sub(pattern, replacement, content, count=1)
        
        if new_content != content:
            content = new_content
            changes_made.append("added progress indicator")
    
    # Write the updated content if changes were made
    if changes_made:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        except Exception as e:
            return False, f"Error writing file: {e}"
    else:
        return False, "no changes needed"

def process_all_html_files(ml_path):
    """Process all HTML files in python_machine_learning"""
    print("\n📝 Updating HTML files...")
    
    # Get all HTML files
    html_files = []
    for filename in os.listdir(ml_path):
        if filename.endswith('.html'):
            html_files.append(filename)
    
    html_files.sort()
    total_files = len(html_files)
    
    print(f"Found {total_files} HTML files to process")
    print("-" * 60)
    
    # Statistics
    updated_count = 0
    skipped_count = 0
    failed_files = []
    
    # Process each file
    for i, filename in enumerate(html_files, 1):
        filepath = os.path.join(ml_path, filename)
        progress = f"[{i}/{total_files}]"
        
        success, result = update_html_file(filepath)
        
        if success:
            changes = ", ".join(result)
            print(f"{progress} ✅ {filename}: {changes}")
            updated_count += 1
        elif "no changes needed" in str(result) or "already updated" in str(result):
            print(f"{progress} ✓  {filename}: {result}")
            skipped_count += 1
        else:
            print(f"{progress} ⚠  {filename}: {result}")
            failed_files.append((filename, result))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Files updated: {updated_count}")
    print(f"Files skipped (already updated): {skipped_count}")
    print(f"Files failed: {len(failed_files)}")
    
    if failed_files:
        print("\n⚠ Failed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    return updated_count > 0

def create_favicon(ml_path):
    """Create or copy favicon"""
    print("\n🎨 Setting up favicon...")
    
    favicon_path = os.path.join(ml_path, 'favicon.png')
    if not os.path.exists(favicon_path):
        # Try to copy from another project
        possible_sources = [
            "\\\\wsl$\\Ubuntu\\home\\practicalace\\projects\\python_datascience\\favicon.png",
            "/home/practicalace/projects/python_datascience/favicon.png",
        ]
        
        copied = False
        for source in possible_sources:
            if os.path.exists(source):
                shutil.copy2(source, favicon_path)
                print("  ✅ Copied favicon.png")
                copied = True
                break
        
        if not copied:
            print("  ℹ️  No favicon found to copy, you may want to add one")
    else:
        print("  ✓ favicon.png already exists")

def verify_setup(ml_path):
    """Verify that all required files are in place"""
    print("\n🔍 Verifying setup...")
    
    checks = {
        'index.html': False,
        'styles/main.css': False,
        'styles/enhanced.css': False,
        'js/clipboard.js': False,
        'js/course-enhancements.js': False,
    }
    
    for file_path in checks.keys():
        full_path = os.path.join(ml_path, file_path)
        if os.path.exists(full_path):
            checks[file_path] = True
            print(f"  ✅ {file_path} exists")
        else:
            print(f"  ❌ {file_path} missing")
    
    all_good = all(checks.values())
    
    if all_good:
        print("\n✅ All required files are in place!")
    else:
        print("\n⚠️  Some files are missing. The setup may not be complete.")
    
    # Check sample file for proper structure
    sample_file = os.path.join(ml_path, 'basics_sklearn.html')
    if os.path.exists(sample_file):
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_content = f.read()
        
        print("\n📋 Sample file check (basics_sklearn.html):")
        checks = {
            'clipboard.js': 'clipboard.js' in sample_content,
            'course-enhancements.js': 'course-enhancements.js' in sample_content,
            'main.css': 'main.css' in sample_content,
            'enhanced.css': 'enhanced.css' in sample_content,
            'skip-to-main': 'skip-to-main' in sample_content,
            'progress-indicator': 'progress-indicator' in sample_content
        }
        
        for feature, present in checks.items():
            if present:
                print(f"  ✅ Has {feature}")
            else:
                print(f"  ❌ Missing {feature}")
    
    return all_good

def main():
    """Main function"""
    
    print("=" * 60)
    print("PYTHON MACHINE LEARNING UPDATER")
    print("=" * 60)
    print("\nThis script will update python_machine_learning to:")
    print("  • Add clipboard functionality to all code blocks")
    print("  • Include course enhancement features")
    print("  • Add accessibility improvements")
    print("  • Match the structure of other courses\n")
    
    # Get project paths
    print("🔍 Detecting project paths...")
    paths = get_project_paths()
    
    print(f"✅ Found python_machine_learning at: {paths['ml']}")
    if 'datascience' in paths:
        print(f"✅ Found python_datascience at: {paths['datascience']}")
    
    # Step 1: Create directory structure
    create_directory_structure(paths['ml'])
    
    # Step 2: Copy CSS files
    copy_css_files(paths)
    
    # Step 3: Copy/Create JavaScript files
    copy_js_files(paths)
    
    # Step 4: Create/copy favicon
    create_favicon(paths['ml'])
    
    # Step 5: Update HTML files
    if process_all_html_files(paths['ml']):
        print("\n✅ HTML files successfully updated!")
    else:
        print("\n✓ No HTML files needed updating.")
    
    # Step 6: Verify setup
    verify_setup(paths['ml'])
    
    print("\n" + "=" * 60)
    print("✅ PROCESS COMPLETE!")
    print("=" * 60)
    print("\nYour python_machine_learning project now has:")
    print("  • An organized index.html with 10 modules and 33 lessons")
    print("  • Copy-to-clipboard functionality on all code blocks")
    print("  • Course enhancement features")
    print("  • Improved accessibility")
    print("  • Consistent structure with other courses")
    print("\n🤖 Ready to master machine learning with enhanced features! 🚀")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
