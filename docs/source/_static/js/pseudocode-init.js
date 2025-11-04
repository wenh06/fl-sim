(function () {
  var CDN = 'https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min';

  function removeCaptionNumber(root) {
    try {
      var n =
        (root && root.querySelector('.psd-caption-number')) ||
        (root && root.querySelector('.pseudocode-caption-number')) ||
        document.querySelector('.psd-caption-number') ||
        document.querySelector('.pseudocode-caption-number');
      if (n) n.remove();
    } catch (e) {
      console.warn('[pseudocode-init] remove number failed:', e);
    }
  }

  function renderAll() {
    try {
      if (!window.pseudocode || typeof window.pseudocode.renderClass !== 'function') {
        throw new Error('pseudocode.js not ready');
      }
      window.pseudocode.renderClass('pseudocode');

      document.querySelectorAll('pre.pseudocode, .pseudocode').forEach(function (el) {
        var root = el.parentNode || document;
        removeCaptionNumber(root);
      });

      if (window.MathJax) {
        if (typeof MathJax.typesetPromise === 'function') MathJax.typesetPromise();
        else if (typeof MathJax.typeset === 'function') MathJax.typeset();
      }
      return true;
    } catch (e) {
      console.warn('[pseudocode-init] renderAll failed:', e);
      return false;
    }
  }

  function whenMathJaxReady(cb) {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
      MathJax.startup.promise.then(cb).catch(function (e) {
        console.warn('[pseudocode-init] MathJax startup failed:', e);
        cb();
      });
    } else cb();
  }

  function ensurePseudocode(cb) {
    if (window.pseudocode && typeof window.pseudocode.renderClass === 'function') {
      cb();
      return;
    }
    if (window.requirejs) {
      requirejs.config({ paths: { 'pseudocode': CDN } });
      requirejs(['pseudocode'], function (pc) {
        window.pseudocode = pc;
        cb();
      }, function (err) {
        console.warn('[pseudocode-init] AMD load failed:', err);
        cb();
      });
    } else cb();
  }

  function onReady(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  onReady(function () {
    var tries = 0;
    (function attempt() {
      ensurePseudocode(function () {
        whenMathJaxReady(function () {
          if (!renderAll() && tries < 50) {
            tries += 1;
            setTimeout(attempt, 100);
          }
        });
      });
    })();
  });
})();
