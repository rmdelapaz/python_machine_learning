#!/usr/bin/env python3
"""
Course Consistency Checker
Validates that all HTML files in the Python Machine Learning course
follow the established style guide and maintain consistency.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import argparse
from collections import defaultdict

class ConsistencyChecker:
    """Check course files for consistency with style guide."""
    
    def __init__(self, course_dir: str = "."):
        self.course_dir = Path(course_dir)
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = {
            'total_files': 0,
            'checked_files': 0,
            'compliant_files': 0,
            'files_with_errors': 0,
            'files_with_warnings': 0
        }
        
        # Required elements for each HTML file
        self.required_elements = {
            'stylesheets': [
                'styles/main.css',
                'styles/enhanced.css'
            ],
            'scripts': [
                'js/course-enhancements.js',
                'js/clipboard.js'
            ],
            'structure': [
                '<a href="#main-content" class="skip-to-main">',
                '<div class="progress-indicator"',
                '<main id="main-content">',
                '<nav class="breadcrumb"',
                '<nav class="lesson-nav"'
            ],
            'meta': [
                '<meta charset="UTF-8">',
                '<meta name="viewport"',
                '<meta name="description"'
            ]
        }
        
        # Title format pattern
        self.title_pattern = r'<title>.*? - Python Machine Learning</title>'
        
        # Old/deprecated patterns to check
        self.deprecated_patterns = {
            'old_stylesheets': [
                'styles/style.css',
                'prism.min.css',
                'prism-tomorrow.min.css'
            ],
            'old_navigation': [
                'Python Data Science',
                'nav-container',
                'nav-logo'
            ]
        }
        
    def check_file(self, filepath: Path) -> Dict:
        """Check a single HTML file for consistency."""
        
        results = {
            'filepath': str(filepath),
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for required stylesheets
            for stylesheet in self.required_elements['stylesheets']:
                if stylesheet not in content:
                    results['errors'].append(f"Missing required stylesheet: {stylesheet}")
                    
            # Check for required scripts
            for script in self.required_elements['scripts']:
                if script not in content:
                    results['warnings'].append(f"Missing script: {script}")
                    
            # Check for required structure elements
            for element in self.required_elements['structure']:
                if element not in content:
                    results['errors'].append(f"Missing required element: {element[:50]}...")
                    
            # Check for meta tags
            for meta in self.required_elements['meta']:
                if meta not in content:
                    results['warnings'].append(f"Missing meta tag: {meta}")
                    
            # Check title format
            if not re.search(self.title_pattern, content):
                results['errors'].append("Title doesn't follow standard format (should end with ' - Python Machine Learning')")
                
            # Check for deprecated patterns
            for old_style in self.deprecated_patterns['old_stylesheets']:
                if old_style in content:
                    results['errors'].append(f"Using deprecated stylesheet: {old_style}")
                    
            for old_nav in self.deprecated_patterns['old_navigation']:
                if old_nav in content:
                    results['warnings'].append(f"Using old navigation pattern: {old_nav}")
                    
            # Check for Mermaid initialization
            if 'mermaid' not in content:
                results['warnings'].append("Mermaid diagram support not initialized")
                
            # Check for breadcrumb navigation
            if '<nav class="breadcrumb"' in content:
                # Verify breadcrumb structure
                breadcrumb_match = re.search(r'<nav class="breadcrumb".*?</nav>', content, re.DOTALL)
                if breadcrumb_match:
                    breadcrumb = breadcrumb_match.group()
                    if 'Home' not in breadcrumb and 'index.html' not in breadcrumb:
                        results['warnings'].append("Breadcrumb should link to Home/index.html")
                        
            # Check for navigation links
            if '<nav class="lesson-nav"' in content:
                nav_match = re.search(r'<nav class="lesson-nav".*?</nav>', content, re.DOTALL)
                if nav_match:
                    nav = nav_match.group()
                    if 'prev-lesson' not in nav and 'Previous' not in nav:
                        results['warnings'].append("Missing previous lesson link")
                    if 'next-lesson' not in nav and 'Next' not in nav:
                        results['warnings'].append("Missing next lesson link")
                    if 'home-link' not in nav and 'index.html' not in nav:
                        results['warnings'].append("Missing home link in navigation")
                        
            # Check for practice exercises
            if '<h2>Practice Exercises</h2>' not in content and '<h3>Exercise' not in content:
                results['suggestions'].append("Consider adding practice exercises")
                
            # Check for key takeaways
            if 'Key Takeaways' not in content and 'takeaway' not in content.lower():
                results['suggestions'].append("Consider adding key takeaways section")
                
            # Check for code blocks
            if '<pre class="language-python">' not in content and '<pre><code' not in content:
                results['suggestions'].append("No code examples found")
                
            # Check file naming convention
            if not self._check_filename_convention(filepath.name):
                results['warnings'].append(f"Filename '{filepath.name}' doesn't follow convention (lowercase with underscores)")
                
        except Exception as e:
            results['errors'].append(f"Error reading file: {str(e)}")
            
        return results
    
    def _check_filename_convention(self, filename: str) -> bool:
        """Check if filename follows the naming convention."""
        # Should be lowercase with underscores, ending in .html
        if not filename.endswith('.html'):
            return True  # Not an HTML file, skip
        
        name = filename[:-5]  # Remove .html
        
        # Check for uppercase letters
        if name != name.lower():
            return False
            
        # Check for hyphens (should use underscores)
        if '-' in name and name != 'index':
            return False
            
        return True
    
    def check_all_files(self) -> None:
        """Check all HTML files in the course directory."""
        
        html_files = list(self.course_dir.glob('*.html'))
        self.stats['total_files'] = len(html_files)
        
        print(f"🔍 Checking {len(html_files)} HTML files for consistency...\n")
        
        for filepath in html_files:
            # Skip index.html as it has different structure
            if filepath.name == 'index.html':
                continue
                
            self.stats['checked_files'] += 1
            results = self.check_file(filepath)
            
            if results['errors']:
                self.errors[filepath.name] = results['errors']
                self.stats['files_with_errors'] += 1
            
            if results['warnings']:
                self.warnings[filepath.name] = results['warnings']
                self.stats['files_with_warnings'] += 1
                
            if not results['errors'] and not results['warnings']:
                self.stats['compliant_files'] += 1
                
            # Print immediate feedback
            self._print_file_results(filepath.name, results)
            
    def _print_file_results(self, filename: str, results: Dict) -> None:
        """Print results for a single file."""
        
        if not results['errors'] and not results['warnings'] and not results['suggestions']:
            print(f"✅ {filename} - Fully compliant")
            return
            
        print(f"\n📄 {filename}")
        
        if results['errors']:
            print("  ❌ Errors:")
            for error in results['errors']:
                print(f"     • {error}")
                
        if results['warnings']:
            print("  ⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"     • {warning}")
                
        if results['suggestions']:
            print("  💡 Suggestions:")
            for suggestion in results['suggestions']:
                print(f"     • {suggestion}")
                
    def generate_report(self, output_file: str = "consistency_report.txt") -> None:
        """Generate a detailed consistency report."""
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PYTHON MACHINE LEARNING COURSE - CONSISTENCY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Statistics
            f.write("STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total HTML files: {self.stats['total_files']}\n")
            f.write(f"Files checked: {self.stats['checked_files']}\n")
            f.write(f"Compliant files: {self.stats['compliant_files']}\n")
            f.write(f"Files with errors: {self.stats['files_with_errors']}\n")
            f.write(f"Files with warnings: {self.stats['files_with_warnings']}\n")
            f.write(f"Compliance rate: {self.stats['compliant_files'] / max(self.stats['checked_files'], 1) * 100:.1f}%\n")
            f.write("\n")
            
            # Files with errors
            if self.errors:
                f.write("FILES WITH ERRORS\n")
                f.write("-" * 40 + "\n")
                for filename, errors in self.errors.items():
                    f.write(f"\n{filename}:\n")
                    for error in errors:
                        f.write(f"  • {error}\n")
                f.write("\n")
            
            # Files with warnings
            if self.warnings:
                f.write("FILES WITH WARNINGS\n")
                f.write("-" * 40 + "\n")
                for filename, warnings in self.warnings.items():
                    f.write(f"\n{filename}:\n")
                    for warning in warnings:
                        f.write(f"  • {warning}\n")
                f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            if self.stats['files_with_errors'] == 0:
                f.write("✅ No critical errors found!\n")
            else:
                f.write(f"❌ {self.stats['files_with_errors']} files need immediate attention.\n")
                
            if self.stats['files_with_warnings'] > 0:
                f.write(f"⚠️  {self.stats['files_with_warnings']} files have warnings to review.\n")
                
            if self.stats['compliant_files'] == self.stats['checked_files']:
                f.write("🎉 All files are fully compliant with the style guide!\n")
                
        print(f"\n📊 Report generated: {output_file}")
        
    def fix_common_issues(self, filepath: Path, dry_run: bool = True) -> List[str]:
        """Attempt to fix common consistency issues automatically."""
        
        fixes_applied = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content = content
                
            # Fix stylesheet references
            if 'styles/style.css' in content:
                content = content.replace('styles/style.css', 'styles/main.css')
                fixes_applied.append("Updated stylesheet reference to styles/main.css")
                
            # Add missing enhanced.css
            if 'styles/main.css' in content and 'styles/enhanced.css' not in content:
                main_css_line = re.search(r'.*styles/main\.css.*\n', content)
                if main_css_line:
                    new_line = main_css_line.group() + '    <link rel="stylesheet" href="styles/enhanced.css">\n'
                    content = content.replace(main_css_line.group(), new_line)
                    fixes_applied.append("Added styles/enhanced.css")
                    
            # Fix title format
            title_match = re.search(r'<title>(.*?)</title>', content)
            if title_match and 'Python Machine Learning' not in title_match.group():
                old_title = title_match.group()
                title_text = title_match.group(1)
                
                # Remove old course names
                title_text = title_text.replace(' - Python Data Science', '')
                title_text = title_text.replace('Python Data Science - ', '')
                
                new_title = f'<title>{title_text} - Python Machine Learning</title>'
                content = content.replace(old_title, new_title)
                fixes_applied.append("Updated title format")
                
            # Add missing skip link
            if 'skip-to-main' not in content and '<body>' in content:
                skip_link = '    <!-- Skip to main content for accessibility -->\n    <a href="#main-content" class="skip-to-main">Skip to main content</a>\n'
                content = content.replace('<body>', '<body>\n' + skip_link)
                fixes_applied.append("Added skip to main content link")
                
            # Add missing progress indicator
            if 'progress-indicator' not in content and 'skip-to-main' in content:
                progress_html = '''    
    <!-- Progress indicator -->
    <div class="progress-indicator" role="progressbar" aria-label="Page scroll progress">
        <div class="progress-bar"></div>
    </div>
'''
                # Insert after skip link
                skip_end = content.find('</a>', content.find('skip-to-main')) + 4
                content = content[:skip_end] + progress_html + content[skip_end:]
                fixes_applied.append("Added progress indicator")
                
            # Save changes if not dry run
            if fixes_applied and not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            fixes_applied.append(f"Error: {str(e)}")
            
        return fixes_applied
    
    def auto_fix_all(self, dry_run: bool = True) -> None:
        """Attempt to fix common issues in all files."""
        
        html_files = list(self.course_dir.glob('*.html'))
        
        mode = "DRY RUN" if dry_run else "APPLYING FIXES"
        print(f"\n🔧 {mode} - Auto-fixing common issues...\n")
        
        total_fixes = 0
        
        for filepath in html_files:
            if filepath.name == 'index.html':
                continue
                
            fixes = self.fix_common_issues(filepath, dry_run)
            
            if fixes:
                print(f"\n📄 {filepath.name}")
                for fix in fixes:
                    print(f"  ✏️  {fix}")
                total_fixes += len(fixes)
                
        if dry_run and total_fixes > 0:
            print(f"\n💡 {total_fixes} fixes identified. Run with --fix to apply them.")
        elif not dry_run:
            print(f"\n✅ {total_fixes} fixes applied.")
            

def main():
    """Main entry point for the consistency checker."""
    
    parser = argparse.ArgumentParser(
        description='Check Python Machine Learning course files for consistency'
    )
    parser.add_argument(
        '--dir', 
        default='.',
        help='Course directory to check (default: current directory)'
    )
    parser.add_argument(
        '--report',
        default='consistency_report.txt',
        help='Output file for detailed report'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix common issues'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    
    args = parser.parse_args()
    
    # Create checker instance
    checker = ConsistencyChecker(args.dir)
    
    # Run auto-fix if requested
    if args.fix or args.dry_run:
        checker.auto_fix_all(dry_run=args.dry_run)
        
    # Always run consistency check
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)
    
    checker.check_all_files()
    
    # Generate report
    checker.generate_report(args.report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = checker.stats['checked_files']
    compliant = checker.stats['compliant_files']
    errors = checker.stats['files_with_errors']
    warnings = checker.stats['files_with_warnings']
    
    print(f"📊 Files checked: {total}")
    print(f"✅ Fully compliant: {compliant} ({compliant/max(total,1)*100:.1f}%)")
    print(f"❌ With errors: {errors}")
    print(f"⚠️  With warnings: {warnings}")
    
    # Exit code based on errors
    exit_code = 0 if errors == 0 else 1
    
    if errors > 0:
        print(f"\n❌ {errors} files need attention. Check the report for details.")
    else:
        print("\n✅ No critical errors found!")
        
    return exit_code
    

if __name__ == "__main__":
    exit(main())