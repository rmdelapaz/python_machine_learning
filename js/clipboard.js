/**
 * Python Machine Learning — clipboard.js
 * Adds a "Copy" button and a small language label to every code block.
 * Output blocks (.card-output) and blocks marked data-nocopy are left alone.
 * Styles live in styles/main.css. No syntax highlighter is involved.
 */
(function () {
    'use strict';

    var LABELS = { python: 'Python', bash: 'Terminal', powershell: 'PowerShell', sql: 'SQL', text: 'Text', json: 'JSON', html: 'HTML', css: 'CSS', javascript: 'JavaScript' };

    function feedback(btn, ok) {
        btn.textContent = ok ? '✓ Copied' : '✗ Failed';
        btn.classList.add(ok ? 'copy-ok' : 'copy-fail');
        window.setTimeout(function () {
            btn.textContent = 'Copy';
            btn.classList.remove('copy-ok', 'copy-fail');
        }, 1800);
    }

    function fallbackCopy(text, btn) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;';
        document.body.appendChild(ta);
        ta.select();
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        feedback(btn, ok);
    }

    function init() {
        document.querySelectorAll('pre > code').forEach(function (code) {
            var pre = code.parentElement;
            if (pre.closest('.card-output') || pre.hasAttribute('data-nocopy') || pre.closest('.code-block-wrapper')) return;
            var wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(pre);

            var lang = (code.className.match(/language-([\w-]+)/) || [])[1];
            if (lang) {
                var tag = document.createElement('span');
                tag.className = 'code-lang';
                tag.setAttribute('aria-hidden', 'true');
                tag.textContent = LABELS[lang] || lang;
                wrapper.appendChild(tag);
            }

            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'copy-button';
            btn.textContent = 'Copy';
            btn.setAttribute('aria-label', 'Copy code to clipboard');
            wrapper.appendChild(btn);
            btn.addEventListener('click', function () {
                var text = code.textContent;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(function () { feedback(btn, true); }, function () { fallbackCopy(text, btn); });
                } else {
                    fallbackCopy(text, btn);
                }
            });
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
