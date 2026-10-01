// Theme toggle and click-to-play video. No YouTube request is made until a visitor asks for a video.
(function () {
  var root = document.documentElement;
  var toggle = document.getElementById('theme');
  if (toggle) {
    toggle.addEventListener('click', function () {
      root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
      try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
    });
  }

  var dialog = document.getElementById('player');
  if (!dialog) return;
  var frame = dialog.querySelector('.frame');

  function close() {
    frame.innerHTML = '';
    if (dialog.open) dialog.close();
  }

  document.querySelectorAll('.shot[data-yt]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var iframe = document.createElement('iframe');
      iframe.src = 'https://www.youtube-nocookie.com/embed/' + btn.dataset.yt + '?autoplay=1&rel=0';
      iframe.title = btn.getAttribute('aria-label') || 'Video';
      iframe.allow = 'autoplay; encrypted-media; picture-in-picture';
      iframe.allowFullscreen = true;
      frame.innerHTML = '';
      frame.appendChild(iframe);
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else window.open('https://www.youtube.com/watch?v=' + btn.dataset.yt, '_blank', 'noopener');
    });
  });

  dialog.querySelector('.close').addEventListener('click', close);
  dialog.addEventListener('close', function () { frame.innerHTML = ''; });
  dialog.addEventListener('click', function (e) { if (e.target === dialog) close(); });
})();

// AI assistant: a floating button opens the chat in a panel. The chat page (agent-chat.html)
// only runs inside this frame and is loaded on first open, so it costs nothing until asked for.
(function () {
  var fab = document.createElement('button');
  fab.id = 'chat-fab';
  fab.type = 'button';
  fab.setAttribute('aria-label', 'Open George AI Assistant');
  fab.innerHTML = '<span class="chat-fab-dot" aria-hidden="true"></span>'
    + '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 3h16a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-7.6l-4.7 3.5A1 1 0 0 1 6 20.7V18H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm0 2v11h4v2.7l3.7-2.7H20V5H4zm3 3h10v2H7V8zm0 3.5h7v2H7v-2z"/></svg>'
    + '<span class="chat-fab-label">Ask my AI</span>';

  var panel = document.createElement('div');
  panel.id = 'chat-panel';
  panel.hidden = true;
  var frame = document.createElement('iframe');
  frame.title = 'George AI Assistant';
  panel.appendChild(frame);

  document.body.appendChild(fab);
  document.body.appendChild(panel);

  var loaded = false;
  var scrollY = 0;
  var narrow = function () { return window.innerWidth <= 767; };

  function open() {
    if (!loaded) { frame.src = 'agent-chat.html'; loaded = true; }
    panel.hidden = false;
    fab.hidden = true;
    if (narrow()) {
      scrollY = window.scrollY;
      document.body.style.cssText += 'position:fixed;top:-' + scrollY + 'px;left:0;right:0;overflow:hidden;';
    }
  }
  function close() {
    if (panel.hidden) return;
    panel.hidden = true;
    fab.hidden = false;
    if (document.body.style.position === 'fixed') {
      ['position', 'top', 'left', 'right', 'overflow'].forEach(function (p) { document.body.style[p] = ''; });
      window.scrollTo(0, scrollY);
    }
  }

  fab.addEventListener('click', open);
  window.addEventListener('message', function (e) { if (e.data === 'close-chat') close(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  document.querySelectorAll('[data-open-chat]').forEach(function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); open(); });
  });
})();
