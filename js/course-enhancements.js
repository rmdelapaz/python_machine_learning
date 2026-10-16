/**
 * Python Machine Learning — course-enhancements.js
 * Theme toggle, Mermaid (theme-aware), scroll progress, table of contents,
 * quiz, mobile menu, keyboard shortcuts, table wrapping, visited-lesson marks.
 *
 * Design rules:
 *  - No dependency on any syntax highlighter (there is none on this site).
 *  - Every feature checks that its elements exist and every storage call is
 *    guarded, so a page that lacks a component never throws.
 */
(function () {
    'use strict';

    var MERMAID_THEMES = {
        light: { primaryColor: '#eef0fd', primaryTextColor: '#1e293b', primaryBorderColor: '#667eea', lineColor: '#586579', secondaryColor: '#f7f8fc', tertiaryColor: '#f1f3f9', fontFamily: "'Segoe UI', system-ui, sans-serif" },
        dark: { primaryColor: '#2e3470', primaryTextColor: '#e2e8f0', primaryBorderColor: '#a5b4fc', lineColor: '#a3b1c6', secondaryColor: '#1e293b', tertiaryColor: '#0f172a', fontFamily: "'Segoe UI', system-ui, sans-serif" }
    };
    var renderCount = 0;

    function store(key, value) {
        try {
            if (value === undefined) return window.localStorage.getItem(key);
            window.localStorage.setItem(key, value);
        } catch (e) { /* private mode or blocked storage: carry on without it */ }
        return null;
    }

    function currentTheme() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }

    /* ---------------- theme ---------------- */
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        var btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.textContent = theme === 'light' ? '🌙' : '☀️';
            btn.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
        }
        renderMermaid();
    }

    function initThemeToggle() {
        var saved = store('theme');
        var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(saved === 'dark' || saved === 'light' ? saved : (prefersDark ? 'dark' : 'light'));
        var btn = document.getElementById('theme-toggle');
        if (!btn) return;
        btn.addEventListener('click', function () {
            var next = currentTheme() === 'light' ? 'dark' : 'light';
            applyTheme(next);
            store('theme', next);
        });
    }

    /* ---------------- Mermaid ----------------
       The lesson <head> imports Mermaid as a module, stores it on window.__mermaid and fires
       'mermaid-ready'. We render from the SOURCE STRING (kept in data-src) so a theme change can
       re-render, and so label text is never re-parsed as HTML. */
    function renderMermaid() {
        var mermaid = window.__mermaid;
        var blocks = document.querySelectorAll('.mermaid');
        if (!mermaid || !blocks.length) return;
        var theme = currentTheme();
        try {
            mermaid.initialize({ startOnLoad: false, theme: 'base', themeVariables: MERMAID_THEMES[theme] });
        } catch (e) { console.error('Mermaid initialize failed: ' + e.message); return; }
        blocks.forEach(function (el) {
            if (!el.dataset.src) el.dataset.src = el.textContent.trim();
            var id = 'mmd-' + (++renderCount);
            Promise.resolve()
                .then(function () { return mermaid.render(id, el.dataset.src); })
                .then(function (out) {
                    el.innerHTML = out.svg;
                    el.setAttribute('data-rendered', theme);
                    el.classList.remove('mermaid-error');
                })
                .catch(function (err) {
                    el.classList.add('mermaid-error');
                    var stray = document.getElementById('d' + id) || document.getElementById(id);
                    if (stray && !el.contains(stray)) stray.remove();
                    console.error('Mermaid diagram failed to render: ' + (err && err.message ? err.message : err));
                });
        });
    }

    /* ---------------- scroll progress ---------------- */
    function initProgressIndicator() {
        var bar = document.querySelector('.progress-bar');
        if (!bar) return;
        var ticking = false;
        function update() {
            ticking = false;
            var docH = document.documentElement.scrollHeight - window.innerHeight;
            bar.style.width = docH > 0 ? Math.min((window.scrollY / docH) * 100, 100) + '%' : '0';
        }
        window.addEventListener('scroll', function () {
            if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
        }, { passive: true });
        update();
    }

    /* ---------------- table of contents ----------------
       Open at the top of the page. Once the reader scrolls past it, it folds into a round button
       (bottom right). Opening it from the button shows a panel under the nav; picking a link
       closes the panel again. */
    function initTOC() {
        var toc = document.querySelector('details.toc-card');
        var links = document.querySelectorAll('.toc-link');
        if (toc) {
            var autoClosed = false;
            var sentinel = document.createElement('div');
            toc.parentNode.insertBefore(sentinel, toc.nextSibling);
            window.addEventListener('scroll', function () {
                if (autoClosed) return;
                var before = sentinel.getBoundingClientRect().top;
                if (before < 60) {
                    autoClosed = true;
                    toc.classList.add('toc-floating');
                    toc.open = false;
                    // The card just left the page flow. Keep the text the reader is looking at where it was
                    // (browsers with scroll anchoring have already done this, so the difference is 0).
                    var shift = sentinel.getBoundingClientRect().top - before;
                    if (shift) window.scrollBy(0, shift);
                }
            }, { passive: true });
            links.forEach(function (a) {
                a.addEventListener('click', function () {
                    if (toc.classList.contains('toc-floating')) toc.open = false;
                });
            });
            document.addEventListener('keydown', function (e) {
                if (e.key === 'Escape' && toc.classList.contains('toc-floating')) toc.open = false;
            });
        }
        if (!links.length || !('IntersectionObserver' in window)) return;
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                links.forEach(function (l) {
                    l.classList.toggle('active', l.getAttribute('href') === '#' + entry.target.id);
                });
            });
        }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });
        links.forEach(function (link) {
            var id = (link.getAttribute('href') || '').slice(1);
            var section = id && document.getElementById(id);
            if (section) observer.observe(section);
        });
    }

    /* ---------------- quiz ---------------- */
    function initQuiz() {
        document.querySelectorAll('.quiz-question').forEach(function (q) {
            var options = q.querySelectorAll('.quiz-option');
            var feedback = q.querySelector('.quiz-feedback');
            if (feedback) feedback.setAttribute('aria-live', 'polite');
            options.forEach(function (opt) {
                opt.setAttribute('type', 'button');
                opt.addEventListener('click', function () {
                    options.forEach(function (o) { o.classList.remove('selected', 'correct', 'incorrect'); });
                    var ok = opt.dataset.correct === 'true';
                    opt.classList.add('selected', ok ? 'correct' : 'incorrect');
                    if (!feedback) return;
                    feedback.textContent = ok
                        ? '✅ Correct! ' + (opt.dataset.explanation || '')
                        : '❌ Not yet. ' + (opt.dataset.hint || 'Try another option.');
                    feedback.className = 'quiz-feedback ' + (ok ? 'correct' : 'incorrect');
                });
            });
        });
    }

    /* ---------------- tables: every table scrolls inside its own box ---------------- */
    function initTableWrap() {
        document.querySelectorAll('main table').forEach(function (table) {
            if (table.closest('.table-wrap')) return;
            var wrap = document.createElement('div');
            wrap.className = 'table-wrap';
            table.parentNode.insertBefore(wrap, table);
            wrap.appendChild(table);
        });
        document.querySelectorAll('.table-wrap').forEach(function (w) {
            if (!w.hasAttribute('tabindex')) { w.setAttribute('tabindex', '0'); w.setAttribute('role', 'region'); w.setAttribute('aria-label', 'Table (scrolls sideways on small screens)'); }
        });
        // A table wider than its box (usually because code cells do not wrap) gets roomier prose columns,
        // so the text beside the code is not squeezed to one word per line. See "table.is-wide" in main.css.
        function markWide() {
            document.querySelectorAll('.table-wrap > table').forEach(function (table) {
                table.classList.remove('is-wide');
                if (table.scrollWidth > table.parentNode.clientWidth + 1) table.classList.add('is-wide');
            });
        }
        document.querySelectorAll('.table-wrap > table td').forEach(function (td) {
            if (!td.classList.contains('num') && td.textContent.trim().length > 24) td.classList.add('cell-long');
        });
        markWide();
        var wideTimer;
        window.addEventListener('resize', function () { clearTimeout(wideTimer); wideTimer = setTimeout(markWide, 150); });
    }

    /* ---------------- mobile menu ---------------- */
    function initMobileMenu() {
        var btn = document.getElementById('mobile-menu-toggle');
        var nav = document.getElementById('nav-links');
        if (!btn || !nav) return;
        function close() { nav.classList.remove('active'); btn.setAttribute('aria-expanded', 'false'); btn.textContent = '☰'; }
        btn.setAttribute('aria-label', 'Menu');
        btn.addEventListener('click', function () {
            var open = nav.classList.toggle('active');
            btn.setAttribute('aria-expanded', String(open));
            btn.textContent = open ? '✕' : '☰';
        });
        document.addEventListener('click', function (e) {
            if (!btn.contains(e.target) && !nav.contains(e.target)) close();
        });
    }

    /* ---------------- keyboard shortcuts ---------------- */
    function showShortcuts() {
        if (document.querySelector('.shortcuts-modal')) return;
        var modal = document.createElement('div');
        modal.className = 'shortcuts-modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-label', 'Keyboard shortcuts');
        modal.innerHTML = '<div class="shortcuts-content"><h3>Keyboard Shortcuts</h3><button type="button" class="close-modal" aria-label="Close">&times;</button>' +
            '<dl><dt>←</dt><dd>Previous lesson</dd><dt>→</dt><dd>Next lesson</dd><dt>H</dt><dd>Course home</dd><dt>T</dt><dd>Toggle dark mode</dd><dt>?</dt><dd>Show this help</dd><dt>Esc</dt><dd>Close</dd></dl></div>';
        document.body.appendChild(modal);
        modal.querySelector('.close-modal').addEventListener('click', function () { modal.remove(); });
        modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });
    }

    function initKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
            var t = e.target;
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
            var link;
            switch (e.key) {
                case 'Escape': { var m = document.querySelector('.shortcuts-modal'); if (m) m.remove(); break; }
                case 'ArrowLeft': link = document.querySelector('.prev-lesson'); if (link) window.location.href = link.href; break;
                case 'ArrowRight': link = document.querySelector('.next-lesson'); if (link) window.location.href = link.href; break;
                case 'h': case 'H': link = document.querySelector('.nav-logo'); if (link) window.location.href = link.href; break;
                case 't': case 'T': { var b = document.getElementById('theme-toggle'); if (b) b.click(); break; }
                case '?': showShortcuts(); break;
            }
        });
    }

    /* ---------------- visited lessons (check marks on the index) ---------------- */
    function initVisited() {
        var visited = {};
        try { visited = JSON.parse(store('dsVisited') || '{}') || {}; } catch (e) { visited = {}; }
        var page = window.location.pathname.split('/').pop() || 'index.html';
        if (document.querySelector('.lesson-nav')) {
            visited[page] = 1;
            store('dsVisited', JSON.stringify(visited));
        }
        document.querySelectorAll('a.lesson-link').forEach(function (a) {
            var href = (a.getAttribute('href') || '').split('#')[0];
            if (visited[href]) a.classList.add('visited');
        });
    }

    function initPrint() {
        var btn = document.getElementById('print-button');
        if (btn) btn.addEventListener('click', function (e) { e.preventDefault(); window.print(); });
        window.addEventListener('beforeprint', function () {
            document.querySelectorAll('details').forEach(function (d) { d.setAttribute('open', ''); });
        });
    }

    function init() {
        var steps = [initThemeToggle, initProgressIndicator, initTableWrap, initTOC, initQuiz, initMobileMenu, initKeyboardShortcuts, initVisited, initPrint];
        steps.forEach(function (fn) {
            try { fn(); } catch (e) { console.error('course-enhancements: ' + fn.name + ' failed: ' + e.message); }
        });
    }

    document.addEventListener('mermaid-ready', renderMermaid);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();

    window.courseEnhancements = { applyTheme: applyTheme, renderMermaid: renderMermaid };
})();
