/* Step-Wise Refusal Dynamics: project page interactions.
   Vanilla JS, no dependencies. Sections:
   1. Mobile nav drawer (ported from Decart Research nav.js)
   2. Hero simulator: the paper's own examples (Figures 1, 2, 11 and 12), replayed step by step
   3. Charts built from the paper's tables (HTML bars, direct labels, tooltips)
   4. SRI explorer (Figures 12 & 13, one model at a time)
   5. Quote word-fill on scroll (Decart home-effects)
   6. Quick-links rail: active section, share
   7. BibTeX copy, reveal-on-scroll, tooltips */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* ------------------------------------------------------------------ */
  /* 1. Mobile navigation drawer                                          */
  /* ------------------------------------------------------------------ */
  (function nav() {
    var toggle = $('.nav-toggle'), links = $('#navLinks');
    if (!toggle || !links) return;
    function setOpen(open) {
      document.body.classList.toggle('nav-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    }
    toggle.addEventListener('click', function () { setOpen(!document.body.classList.contains('nav-open')); });
    links.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  })();

  /* ------------------------------------------------------------------ */
  /* 2. Hero simulator: schematic of recovery-by-revision (diffusion vs AR) */
  /* ------------------------------------------------------------------ */
  /* The card illustrates the mechanism described in Sections 2.2 and 3 of the
     paper (Figures 1, 2 and 12). Token sequences and curves are hand-authored
     and clearly captioned as a schematic; nothing here is measured data.
     Digitized curves from the paper's figures live in assets/data/sri-real.json
     for anyone who wants real trajectories. */
  (function sim() {
    var tokensEl = $('#simTokens'), stateEl = $('#simState'), stepEl = $('#simStep'),
        svg = $('#simSri'), sriVal = $('#simSriVal'), play = $('#simPlay'),
        slider = $('#simSlider'), fill = $('#simFill'), caption = $('#simCaption'),
        modes = $('#simModes');
    if (!tokensEl || !svg) return;

    var T = 32, SLOTS = 16;
    var RED = '▇▇▇▇'; /* redacted block glyphs */

    /* Diffusion: per slot, a list of [step, kind, text, committed]. Harmful predictions
       appear early and stay uncommitted; refusal tokens are committed from step 12 on. */
    var DIFF = [
      [[2, 'harmful', 'Sure,', 0], [21, 'revising', "I'm", 0], [23, 'safe', "I'm", 1]],
      [[4, 'harmful', 'here is', 0], [10, 'revising', 'sorry,', 0], [12, 'safe', 'sorry,', 1]],
      [[1, 'harmful', 'how', 0], [13, 'revising', 'but', 0], [14, 'safe', 'but', 1]],
      [[6, 'harmful', 'to', 0], [16, 'revising', 'I', 0], [17, 'safe', 'I', 1]],
      [[3, 'harmful', 'make', 0], [15, 'safe', "can't", 1]],
      [[7, 'harmful', RED, 0], [18, 'revising', 'assist', 0], [19, 'safe', 'assist', 1]],
      [[5, 'harmful', RED.slice(0, 3), 0], [20, 'safe', 'with', 1]],
      [[9, 'harmful', 'the', 0], [22, 'safe', 'that', 1]],
      [[8, 'harmful', RED + '▇', 0], [24, 'revising', 'request.', 0], [25, 'safe', 'request.', 1]],
      [[11, 'harmful', RED.slice(0, 2), 0], [26, 'eos', '', 1]],
      [[10, 'harmful', RED + '▇▇', 0], [27, 'eos', '', 1]],
      [[12, 'harmful', 'should', 0], [28, 'eos', '', 1]],
      [[13, 'harmful', 'be', 0], [29, 'eos', '', 1]],
      [[14, 'harmful', RED.slice(0, 3), 0], [30, 'eos', '', 1]],
      [[15, 'harmful', RED, 0], [31, 'eos', '', 1]],
      [[16, 'harmful', 'the', 0], [32, 'eos', '', 1]]
    ];
    /* AR: one token committed every two steps, left to right, never revised. */
    var AR_TOKENS = ['Sure,', 'here is', 'how', 'to', 'make', RED, RED.slice(0, 3), 'the', RED + '▇', RED.slice(0, 2), RED + '▇▇', 'should', 'be', RED.slice(0, 3), RED, 'the'];

    /* Schematic SRI trajectories (sigma_t: 1 = compliance-aligned, 0 = refusal-aligned). */
    var SRI = {
      diffusion: [0.92, 0.93, 0.94, 0.95, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.80, 0.74, 0.66, 0.60, 0.55, 0.50, 0.46, 0.42, 0.38, 0.34, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09],
      ar: [0.90, 0.91, 0.92, 0.93, 0.92, 0.88, 0.80, 0.72, 0.66, 0.70, 0.62, 0.58, 0.64, 0.55, 0.60, 0.52, 0.57, 0.50, 0.55, 0.48, 0.53, 0.50, 0.46, 0.52, 0.47, 0.50, 0.45, 0.49, 0.47, 0.50, 0.48, 0.51, 0.49],
      harmless: [], refusal: []
    };
    for (var i = 0; i <= T; i++) { SRI.harmless.push(0.95 + 0.012 * Math.sin(i * 1.3)); SRI.refusal.push(0.08 + 0.008 * Math.cos(i * 0.9)); }

    var CAPTIONS = {
      diffusion: '<strong>Diffusion remasking.</strong> Every step predicts all positions and commits only the positions selected by the remasking rule. Harmful predictions (dashed red) compete for commitment and in this example stay uncommitted, and once refusal tokens are committed (solid green) the rest of the response follows them. This card is a schematic of the mechanism shown in Figures 1, 2 and 12 of the paper, with harmful content redacted; the curves are illustrative and do not reproduce measured data.',
      ar: '<strong>Autoregressive sampling.</strong> Each token is committed the moment it is generated. Once the compliance prefix is out, the harmful continuation cannot be revised, and the SRI signal never settles into the refusal zone. This is the incomplete internal recovery that the final text alone would hide. The card is a schematic and does not reproduce measured data.'
    };

    function stateFor(mode, step) {
      if (step === 0) return { kind: 'mask', label: 'Fully masked' };
      if (mode === 'diffusion') {
        if (step < 12) return { kind: 'harmful', label: 'Harmful predictions forming · none committed' };
        if (step < 25) return { kind: 'revising', label: 'Revising · refusal tokens being committed' };
        return { kind: 'safe', label: 'Recovered · safe final output' };
      }
      if (step < 5) return { kind: 'harmful', label: 'Compliance prefix committed' };
      if (step < 32) return { kind: 'harmful', label: 'Harmful tokens committed · cannot be revised' };
      return { kind: 'harmful', label: 'Jailbreak succeeded · no recovery' };
    }

    function tokensAt(mode, step) {
      var out = [];
      for (var s = 0; s < SLOTS; s++) {
        var tok = { kind: 'mask', text: '', committed: false, changedAt: -1 };
        if (mode === 'diffusion') {
          var evs = DIFF[s];
          for (var k = 0; k < evs.length; k++) {
            if (evs[k][0] <= step) { tok = { kind: evs[k][1], text: evs[k][2], committed: !!evs[k][3], changedAt: evs[k][0] }; }
          }
        } else {
          var commitStep = 2 * s + 1;
          if (commitStep <= step) tok = { kind: 'harmful', text: AR_TOKENS[s], committed: true, changedAt: commitStep };
        }
        out.push(tok);
      }
      return out;
    }

    /* Build the 16 token nodes once; then mutate classes and text per step. */
    var nodes = [];
    for (var n = 0; n < SLOTS; n++) { var d = el('span', 'tok tok--mask', '░░░'); tokensEl.appendChild(d); nodes.push(d); }

    function renderTokens(mode, step) {
      var toks = tokensAt(mode, step);
      for (var i = 0; i < SLOTS; i++) {
        var t = toks[i], node = nodes[i];
        var cls = 'tok tok--' + t.kind + (t.committed ? ' is-committed' : '') + (t.changedAt === step ? ' is-new' : '');
        var text = t.kind === 'mask' ? '░░░' : (t.kind === 'eos' ? '<eos>' : t.text);
        if (node.className !== cls) node.className = cls;
        if (node.textContent !== text) node.textContent = text;
      }
    }

    /* SRI sparkline */
    var W = 320, H = 112, PX0 = 26, PX1 = 312, PY0 = 12, PY1 = 92;
    var ns = 'http://www.w3.org/2000/svg';
    function sx(t) { return PX0 + (PX1 - PX0) * t / T; }
    function sy(v) { return PY0 + (PY1 - PY0) * (1 - v); }
    function mk(tag, attrs) { var e = document.createElementNS(ns, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); return e; }
    function pathFor(arr, upto) {
      var dstr = '';
      for (var i = 0; i <= upto; i++) dstr += (i ? ' L' : 'M') + sx(i).toFixed(1) + ' ' + sy(arr[i]).toFixed(1);
      return dstr;
    }
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.appendChild(mk('rect', { x: PX0, y: PY0, width: PX1 - PX0, height: (PY1 - PY0) / 2, 'class': 'zone-comp' }));
    svg.appendChild(mk('rect', { x: PX0, y: PY0 + (PY1 - PY0) / 2, width: PX1 - PX0, height: (PY1 - PY0) / 2, 'class': 'zone-ref' }));
    [0, 0.5, 1].forEach(function (v) { svg.appendChild(mk('line', { x1: PX0, x2: PX1, y1: sy(v), y2: sy(v), 'class': 'grid' })); });
    var zl1 = mk('text', { x: PX0 + 6, y: PY0 + 10, 'class': 'zone-label' }); zl1.textContent = 'compliance-aligned'; svg.appendChild(zl1);
    var zl2 = mk('text', { x: PX0 + 6, y: PY1 - 4, 'class': 'zone-label' }); zl2.textContent = 'refusal-aligned'; svg.appendChild(zl2);
    [['1', 1], ['.5', 0.5], ['0', 0]].forEach(function (p) { var tx = mk('text', { x: PX0 - 5, y: sy(p[1]) + 3, 'class': 'axis-label', 'text-anchor': 'end' }); tx.textContent = p[0]; svg.appendChild(tx); });
    [0, 8, 16, 24, 32].forEach(function (s) { var tx = mk('text', { x: sx(s), y: H - 4, 'class': 'axis-label', 'text-anchor': 'middle' }); tx.textContent = s; svg.appendChild(tx); });
    svg.appendChild(mk('path', { d: pathFor(SRI.harmless, T), 'class': 'ref-harmless' }));
    svg.appendChild(mk('path', { d: pathFor(SRI.refusal, T), 'class': 'ref-refusal' }));
    var ghost = mk('path', { d: '', 'class': 'path-ghost' }); svg.appendChild(ghost);
    var main = mk('path', { d: '', 'class': 'path-main' }); svg.appendChild(main);
    var marker = mk('circle', { r: 4.5, cx: sx(0), cy: sy(0.92), 'class': 'marker' }); svg.appendChild(marker);

    function renderSri(mode, step) {
      var arr = SRI[mode];
      ghost.setAttribute('d', pathFor(arr, T));
      main.setAttribute('d', pathFor(arr, step));
      marker.setAttribute('cx', sx(step)); marker.setAttribute('cy', sy(arr[step]));
      sriVal.textContent = 'σ = ' + arr[step].toFixed(2);
    }

    var mode = 'diffusion', step = 0, timer = null, playing = false;

    function render() {
      renderTokens(mode, step);
      renderSri(mode, step);
      var st = stateFor(mode, step);
      stateEl.setAttribute('data-kind', st.kind);
      stateEl.lastElementChild.textContent = st.label;
      stepEl.textContent = step;
      slider.value = step;
      fill.style.width = (step / T * 100) + '%';
    }
    function setStep(s) { step = Math.max(0, Math.min(T, s)); render(); }
    function setMode(m) {
      mode = m;
      $$('button', modes).forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute('data-mode') === m ? 'true' : 'false'); });
      caption.innerHTML = CAPTIONS[m];
      setStep(0);
    }

    var holdTicks = 0;
    function tick() {
      if (step >= T) { holdTicks++; if (holdTicks > 6) { holdTicks = 0; setStep(0); } return; }
      setStep(step + 1);
    }
    function setPlaying(p) {
      playing = p;
      play.classList.toggle('is-playing', p);
      play.setAttribute('aria-label', p ? 'Pause animation' : 'Play animation');
      if (timer) { clearInterval(timer); timer = null; }
      if (p) timer = setInterval(tick, 300);
    }

    play.addEventListener('click', function () { setPlaying(!playing); });
    slider.addEventListener('input', function () { setPlaying(false); setStep(parseInt(slider.value, 10)); });
    modes.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-mode]'); if (!b) return;
      var wasPlaying = playing; setMode(b.getAttribute('data-mode')); if (wasPlaying || !reduced) setPlaying(true);
    });

    setMode(/[?&]mode=ar\b/.test(location.search) ? 'ar' : 'diffusion');
    if (!reduced) {
      setPlaying(true);
      /* pause while off-screen so the animation is fresh when the reader returns */
      if ('IntersectionObserver' in window) {
        var wasAuto = true;
        new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (!en.isIntersecting && playing) { setPlaying(false); wasAuto = true; }
            else if (en.isIntersecting && !playing && wasAuto) { setPlaying(true); }
          });
        }, { threshold: 0.15 }).observe($('#heroDemo'));
        play.addEventListener('click', function () { wasAuto = false; });
      }
    }
  })();

  /* ------------------------------------------------------------------ */
  /* 3. Charts                                                            */
  /* ------------------------------------------------------------------ */
  var tip = $('#chartTip');
  function bindTips(root) {
    $$('[data-tip]', root).forEach(function (n) {
      n.addEventListener('pointerenter', function (e) { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); moveTip(e); });
      n.addEventListener('pointermove', moveTip);
      n.addEventListener('pointerleave', function () { tip.classList.remove('is-on'); });
      n.addEventListener('focus', function () { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); var r = n.getBoundingClientRect(); tip.style.left = (r.left + r.width / 2) + 'px'; tip.style.top = r.top + 'px'; });
      n.addEventListener('blur', function () { tip.classList.remove('is-on'); });
    });
  }
  function moveTip(e) { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }

  /* Horizontal bar renderer.
     groups: [{label, sub, isBreak, bars:[{value, cls, text, tip}]}], max: axis max */
  function renderHBars(container, groups, max, ticks) {
    container.innerHTML = '';
    var frag = document.createDocumentFragment();
    groups.forEach(function (g) {
      var row = el('div', 'hbar-group' + (g.isBreak ? ' is-family-break' : ''));
      var lab = el('div', 'hbar-label'); lab.appendChild(document.createTextNode(g.label));
      if (g.sub) lab.appendChild(el('small', null, g.sub));
      row.appendChild(lab);
      var set = el('div', 'hbar-track-set');
      g.bars.forEach(function (b) {
        var r = el('div', 'hbar-row'); r.setAttribute('tabindex', '0');
        if (b.tip) r.setAttribute('data-tip', b.tip);
        r.appendChild(el('span', 'hbar-baseline'));
        var bar = el('div', 'bar' + (b.cls ? ' ' + b.cls : '')); bar.style.width = '0%'; bar.setAttribute('data-w', Math.max(0, Math.min(100, b.value / max * 100)));
        r.appendChild(bar);
        var v = el('span', 'val'); v.innerHTML = b.text; r.appendChild(v);
        set.appendChild(r);
      });
      row.appendChild(set);
      frag.appendChild(row);
    });
    if (ticks) {
      var gr = el('div', 'hbar-group');
      gr.appendChild(el('div'));
      var grid = el('div', 'hbar-grid');
      ticks.forEach(function (t, i) { var s = el('span', i === 0 ? 'is-first' : (i === ticks.length - 1 ? 'is-last' : null), t.label); s.style.left = (t.value / max * 100) + '%'; grid.appendChild(s); });
      gr.appendChild(grid); frag.appendChild(gr);
    }
    container.appendChild(frag);
    bindTips(container);
    requestAnimationFrame(function () { $$('.bar', container).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
  }

  /* --- Table 1: HRR / FRR --- */
  (function hrr() {
    var c = $('#chartHrr'); if (!c) return;
    var rows = [['LLaDA', 0.81, 0.63], ['Dream', 0.96, 0.73], ['LLaDA-1.5', 0.92, 0.65]];
    renderHBars(c, rows.map(function (r) {
      return { label: r[0], sub: 'diffusion', bars: [
        { value: r[1], text: r[1].toFixed(2), tip: r[0] + ' · Harmful Remasking Rate ' + r[1].toFixed(2) },
        { value: r[2], cls: 'is-ar', text: r[2].toFixed(2), tip: r[0] + ' · Full Recovery Rate ' + r[2].toFixed(2) }
      ] };
    }), 1, [{ value: 0, label: '0' }, { value: 0.5, label: '0.5' }, { value: 1, label: '1.0' }]);
  })();

  /* --- Table 2: delta RR / delta ASR (AR -> diffusion, fixed weights) --- */
  (function delta() {
    var c = $('#chartDelta'), seg = $('#deltaMetric'); if (!c) return;
    var attacks = ['WildJailbreak', 'Flip Attack', 'PAIR', 'Refusal Suppression', 'Random Search'];
    var data = { // [LLaDA, LLaDA-1.5]
      rr: [[14, 15], [33, 38], [7, 0], [55, 52], [21, 19]],
      asr: [[18, 21], [36, 60], [5, 2], [26, 20], [16, 8]]
    };
    var deltaNote = $('#deltaNote');
    var NOTES = {
      rr: 'Every value is non-negative; PAIR moves least and Refusal Suppression most in refusal rate. Greedy, random, static and dynamic remasking variants all improve on AR sampling (Appendix A), and an LLM judge confirms every refusal-rate gain (Table 15).',
      asr: 'Every value is non-negative; PAIR moves least and Flip Attack most in attack success. Greedy, random, static and dynamic remasking variants all improve on AR sampling (Appendix A).'
    };
    function draw(metric) {
      if (deltaNote) deltaNote.textContent = NOTES[metric];
      var name = metric === 'rr' ? 'Δ Refusal Rate' : 'Δ Attack Success Rate (reduction)';
      renderHBars(c, attacks.map(function (a, i) {
        var v = data[metric][i];
        return { label: a, bars: [
          { value: v[0], text: '+' + v[0], tip: 'LLaDA · ' + a + ' · ' + name + ' +' + v[0] + ' points' },
          { value: v[1], cls: 'is-ar', text: '+' + v[1], tip: 'LLaDA-1.5 · ' + a + ' · ' + name + ' +' + v[1] + ' points' }
        ] };
      }), 60, [{ value: 0, label: '0' }, { value: 20, label: '+20' }, { value: 40, label: '+40' }, { value: 60, label: '+60' }]);
    }
    seg.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-metric]'); if (!b) return;
      $$('button', seg).forEach(function (x) { x.setAttribute('aria-selected', x === b ? 'true' : 'false'); });
      draw(b.getAttribute('data-metric'));
    });
    draw('rr');
  })();

  /* --- Table 3: AR vs diffusion across attacks --- */
  (function gap() {
    var c = $('#chartGap'), seg = $('#gapMetric'), sel = $('#gapAttack'), btn = $('#gapTableBtn'), tv = $('#gapTableView'), note = $('#gapNote');
    if (!c) return;
    var cols = ['raw', 'all', 'flip', 'pair', 'refsup', 'random', 'wild'];
    var colNames = { raw: 'Raw harmful prompts', all: 'All jailbreaks', flip: 'Flip Attack', pair: 'PAIR', refsup: 'Refusal Suppression', random: 'Random Search', wild: 'WildJailbreak' };
    var models = [ // name, family, then per column [RR, ASR]
      ['LLaMA-3', 'ar', [83, 13], [23.4, 59.2], [1, 98], [70, 24], [41, 48], [3, 29], [2, 97]],
      ['Qwen-2.5', 'ar', [53, 26], [11.4, 62.2], [2, 91], [33, 44], [18, 46], [4, 31], [0, 99]],
      ['Gemma', 'ar', [88, 8], [46.2, 48.2], [10, 86], [76, 20], [74, 17], [65, 26], [6, 92]],
      ['LLaDA', 'dlm', [83, 6], [67.4, 18.4], [69, 24], [92, 5], [79, 8], [36, 26], [61, 29]],
      ['LLaDA-1.5', 'dlm', [84, 9], [59.6, 21.0], [59, 17], [92, 2], [71, 15], [27, 34], [49, 37]],
      ['Dream', 'dlm', [89, 1], [44.4, 9.4], [42, 18], [86, 0], [46, 5], [34, 4], [14, 20]]
    ];
    var metric = 'asr', attack = 'all';
    function fmt(v) { return (Math.round(v * 10) / 10) + '%'; }
    function draw() {
      var ci = cols.indexOf(attack) + 2, mi = metric === 'rr' ? 0 : 1;
      var arV = [], dlmV = [];
      renderHBars(c, models.map(function (m, idx) {
        var v = m[ci][mi];
        (m[1] === 'ar' ? arV : dlmV).push(v);
        return { label: m[0], sub: m[1] === 'ar' ? 'autoregressive' : 'diffusion', isBreak: idx === 3,
          bars: [{ value: v, cls: m[1] === 'ar' ? 'is-ar' : '', text: fmt(v), tip: m[0] + ' · ' + colNames[attack] + ' · ' + (metric === 'rr' ? 'Refusal Rate ' : 'Attack Success Rate ') + fmt(v) }] };
      }), 100, [{ value: 0, label: '0' }, { value: 25, label: '25' }, { value: 50, label: '50' }, { value: 75, label: '75' }, { value: 100, label: '100%' }]);
      var mn = function (a) { return Math.min.apply(null, a); }, mx = function (a) { return Math.max.apply(null, a); };
      var lab = metric === 'rr' ? 'Refusal Rate' : 'Attack Success Rate';
      note.innerHTML = '<strong>' + colNames[attack] + ' · ' + lab + '.</strong> Autoregressive models: ' + fmt(mn(arV)) + ' to ' + fmt(mx(arV)) + '. Diffusion models: ' + fmt(mn(dlmV)) + ' to ' + fmt(mx(dlmV)) + '.' +
        (attack === 'raw' ? ' On raw harmful prompts all models exhibit relatively high refusal rates; the gap emerges under jailbreak attacks.' : (metric === 'asr' ? ' Lower is safer.' : ' Higher is safer.'));
    }
    seg.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-metric]'); if (!b) return;
      $$('button', seg).forEach(function (x) { x.setAttribute('aria-selected', x === b ? 'true' : 'false'); });
      metric = b.getAttribute('data-metric'); draw();
    });
    sel.addEventListener('change', function () { attack = sel.value; draw(); });
    // full table view
    (function buildTable() {
      var t = el('table', 'data-table'), thead = el('thead'), tr = el('tr');
      tr.appendChild(el('th', null, 'Model'));
      cols.forEach(function (k) { tr.appendChild(el('th', null, colNames[k] + ' RR ↑')); tr.appendChild(el('th', null, colNames[k] + ' ASR ↓')); });
      thead.appendChild(tr); t.appendChild(thead);
      var tb = el('tbody');
      models.forEach(function (m) {
        var r = el('tr'); var first = el('td', null, m[0] + (m[1] === 'ar' ? ' (AR)' : ' (diffusion)')); r.appendChild(first);
        for (var i = 2; i < m.length; i++) { r.appendChild(el('td', m[1] === 'ar' ? 'is-ar' : '', fmt(m[i][0]))); r.appendChild(el('td', m[1] === 'ar' ? 'is-ar' : '', fmt(m[i][1]))); }
        tb.appendChild(r);
      });
      t.appendChild(tb);
      var cap = el('caption', null, 'Table 3 of the paper. RR: Refusal Rate (higher is safer). ASR: Attack Success Rate (lower is safer).');
      t.appendChild(cap);
      tv.appendChild(t);
    })();
    btn.addEventListener('click', function () {
      var on = btn.getAttribute('aria-pressed') !== 'true';
      btn.setAttribute('aria-pressed', on ? 'true' : 'false'); tv.hidden = !on; btn.textContent = on ? 'Hide table' : 'Full table';
    });
    draw();
  })();

  /* --- Table 4: ablation AUROC --- */
  (function ablation() {
    var c = $('#chartAblation'); if (!c) return;
    var rows = [['Text-based signal', 0.573, 0.221], ['Static activations (first step)', 0.584, 0.138], ['SRI, first layer', 0.566, 0.101], ['SRI, middle layer', 0.864, 0.062], ['SRI, last layer (ours)', 0.920, 0.047]];
    renderHBars(c, rows.map(function (r, i) {
      var ours = i === rows.length - 1;
      return { label: r[0], bars: [{ value: r[1], cls: ours ? 'is-emph' : 'is-dim', text: r[1].toFixed(3) + ' <small>± ' + r[2].toFixed(3) + '</small>', tip: r[0] + ' · AUROC ' + r[1].toFixed(3) + ' ± ' + r[2].toFixed(3) }] };
    }), 1, [{ value: 0, label: '0' }, { value: 0.5, label: '0.5' }, { value: 1, label: '1.0' }]);
  })();

  /* --- Table 13: defenses per model --- */
  (function defenses() {
    var rowsEl = $('#defenseRows'), sel = $('#defenseModel'), note = $('#defenseNote'); if (!rowsEl) return;
    var D = ['Undefended', 'PPL filtering', 'Self-Examine', 'LlamaGuard 3', 'SRI Guard'];
    var data = { // per model: rows of [overhead %, FP %, RR %, ASR %]
      llada:   [[0.00, 7, 67.4, 18.4], [6.26, 9, 68.0, 18.2], [7.74, 7, 67.4, 18.4], [12.42, 7, 77.2, 14.4], [0.04, 9, 73.4, 16.8]],
      llada15: [[0.00, 6, 59.6, 21.0], [6.18, 8, 60.2, 20.8], [8.09, 6, 80.2, 10.0], [12.27, 7, 71.6, 16.6], [0.04, 8, 70.2, 17.0]],
      dream:   [[0.00, 4, 44.4, 9.4], [5.72, 6, 44.8, 9.4], [5.33, 4, 47.4, 8.8], [11.34, 4, 51.4, 8.6], [0.03, 6, 56.4, 7.2]],
      qwen25:  [[0.00, 0, 11.4, 62.2], [2.40, 2, 12.0, 59.2], [3.71, 0, 11.4, 62.2], [4.76, 0, 43.2, 46.6], [0.01, 3, 47.8, 40.6]],
      llama3:  [[0.00, 0, 23.4, 59.2], [2.36, 2, 24.0, 58.8], [4.80, 0, 30.8, 44.4], [4.67, 0, 46.6, 48.0], [0.01, 4, 55.0, 43.6]],
      gemma:   [[0.00, 0, 46.2, 48.2], [2.65, 2, 46.8, 47.8], [5.47, 0, 53.4, 37.0], [5.26, 2, 60.4, 37.6], [0.02, 0, 54.2, 42.2]]
    };
    var names = { llada: 'LLaDA', llada15: 'LLaDA-1.5', dream: 'Dream', qwen25: 'Qwen-2.5', llama3: 'LLaMA-3', gemma: 'Gemma' };
    var maxes = [13, 10, 100, 100], labels = ['Overhead', 'False positives', 'Jailbreak refusal', 'Attack success'];
    function fmtv(i, v) { return i === 0 ? v.toFixed(2) + '%' : (i === 1 ? v.toFixed(0) + '%' : v.toFixed(1) + '%'); }
    function draw(model) {
      rowsEl.innerHTML = '';
      var rows = data[model];
      rows.forEach(function (r, i) {
        var row = el('div', 'defense-row' + (i === 4 ? ' is-ours' : ''));
        var nm = el('div', 'defense-name'); nm.appendChild(document.createTextNode(D[i])); if (i === 4) nm.appendChild(el('small', null, 'ours · internal signal')); row.appendChild(nm);
        r.forEach(function (v, j) {
          var m = el('div', 'mini'); m.setAttribute('tabindex', '0');
          m.setAttribute('data-tip', D[i] + ' on ' + names[model] + ' · ' + labels[j] + ' ' + fmtv(j, v));
          var tr = el('div', 'track'); var f = el('div', 'fill'); f.setAttribute('data-w', Math.min(100, v / maxes[j] * 100)); tr.appendChild(f); m.appendChild(tr);
          var num = el('span', 'num', fmtv(j, v)); num.setAttribute('data-label', labels[j]); m.appendChild(num);
          row.appendChild(m);
        });
        rowsEl.appendChild(row);
      });
      bindTips(rowsEl);
      requestAnimationFrame(function () { $$('.fill', rowsEl).forEach(function (f) { f.style.width = f.getAttribute('data-w') + '%'; }); });
      var lg = rows[3], sri = rows[4];
      note.innerHTML = '<strong>' + names[model] + '.</strong> SRI Guard adds ' + sri[0].toFixed(2) + '% overhead against ' + lg[0].toFixed(2) + '% for LlamaGuard 3, two orders of magnitude less, and reaches a jailbreak refusal rate of ' + sri[2].toFixed(1) + '% against ' + lg[2].toFixed(1) + '% for LlamaGuard 3. The detection threshold is the 99% quantile of the reconstruction error on held-out harmless prompts; the false positives of the undefended model come from the base model refusing benign prompts on its own.';
    }
    sel.addEventListener('change', function () { draw(sel.value); });
    draw('llada');
  })();

  /* ------------------------------------------------------------------ */
  /* 4. SRI explorer                                                      */
  /* ------------------------------------------------------------------ */
  (function explorer() {
    var tabs = $('#explorerTabs'), sriImg = $('#explorerSri'), ldaImg = $('#explorerLda'), note = $('#explorerNote'), fam = $('#explorerFam');
    if (!tabs) return;
    var INFO = {
      llama3:  { name: 'LLaMA-3', fam: 'Autoregressive', note: 'The harmless and refusal references stay flat and well separated. The jailbreak trajectory drops into the refusal region within the first few steps and then climbs back into the compliance region, an internal move toward refusal that does not complete while the text-level signal stays flat. LLaMA-3 reaches a 59.2% Attack Success Rate across jailbreaks, its Internal Recovery Rate stays near 0.1 at every threshold (Figure 5), and SRI Guard lifts its jailbreak refusal rate from 23.4% to 55.0% at 0.01% overhead (Table 13).' },
      qwen25:  { name: 'Qwen-2.5', fam: 'Autoregressive', note: 'This is the model behind Figure 2 of the paper: the harmful generation descends toward the refusal region, hovers around the 0.5 boundary between steps 5 and 13, and drifts back toward compliance while the text keeps complying. Qwen-2.5 has the highest jailbreak Attack Success Rate of the six models (62.2%); SRI Guard lowers it to 40.6% and raises the refusal rate from 11.4% to 47.8% (Table 13).' },
      gemma:   { name: 'Gemma', fam: 'Autoregressive', note: 'Gemma refuses jailbreaks about as often as Dream (46.2% against 44.4%), yet its Attack Success Rate is 48.2% against 9.4% for Dream, so a refusal rate alone does not describe robustness. Its jailbreak SRI signal dips below 0.5 in the first steps and then stays volatile inside the compliance region for the whole generation.' },
      llada:   { name: 'LLaDA', fam: 'Diffusion', note: 'LLaDA is the primary diffusion baseline. Harmful intermediate content is revised as remasking proceeds (HRR 0.81, FRR 0.63), and its Internal Recovery Rate stays far above the autoregressive models at every threshold (Figure 5). Running the same weights with AR sampling lowers IRR by up to 0.46 (Figure 6). In this jailbreak example the signal stays between 0.4 and 0.7 for the first eighteen steps, crossing the 0.5 boundary between steps 8 and 13, before rising into the compliance region and ending near 0.8.' },
      llada15: { name: 'LLaDA-1.5', fam: 'Diffusion', note: 'LLaDA-1.5 is a newer LLaDA variant with the same qualitative behavior: HRR 0.92, FRR 0.65, and an aggregate jailbreak Attack Success Rate of 21.0% against 48 to 62% for the autoregressive models. Under AR sampling of the same weights, IRR falls by up to 0.48 (Figure 6). Its jailbreak trajectory hovers between 0.4 and 0.67 for the first fourteen steps and ends near 0.75, well below the flat harmless reference.' },
      dream:   { name: 'Dream', fam: 'Diffusion', note: 'Dream is a different diffusion architecture with the strongest recovery statistics of the three (HRR 0.96, FRR 0.73) and the lowest jailbreak Attack Success Rate in the study, 9.4%. Its jailbreak trajectory hovers around the 0.5 boundary, far from the flat harmless reference, for the first sixteen steps, then rises to about 0.95 and runs alongside the harmless reference for the rest of the generation.' }
    };
    function show(key) {
      $$('button', tabs).forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute('data-model') === key ? 'true' : 'false'); });
      var info = INFO[key];
      sriImg.src = 'assets/figures/fig12-sri-' + key + '.png';
      sriImg.alt = 'SRI signal over 32 generation steps for ' + info.name + ' under a jailbreak prompt, with flat harmless and refusal reference trajectories and shaded compliance and refusal regions.';
      ldaImg.src = 'assets/figures/fig13-lda-' + key + '.png';
      ldaImg.alt = 'Two-dimensional LDA projection of the SRI space for ' + info.name + ': harmless, harmful and refusal responses occupy distinct but partly overlapping regions.';
      fam.textContent = info.fam;
      note.innerHTML = '<strong>' + info.name + '.</strong> ' + info.note;
    }
    tabs.addEventListener('click', function (e) { var b = e.target.closest('button[data-model]'); if (b) show(b.getAttribute('data-model')); });
    show('llama3');
  })();

  /* ------------------------------------------------------------------ */
  /* 5. Quote word-fill on scroll                                          */
  /* ------------------------------------------------------------------ */
  (function quote() {
    var q = $('#quoteText'); if (!q) return;
    var words = [], text = q.textContent; q.textContent = '';
    text.split(/(\s+)/).forEach(function (part) {
      if (/^\s+$/.test(part)) q.appendChild(document.createTextNode(part));
      else if (part.length) { var w = el('span', 'qw', part); q.appendChild(w); words.push(w); }
    });
    function update() {
      var r = q.getBoundingClientRect(), vh = window.innerHeight;
      var p = Math.min(1, Math.max(0, (vh * 0.85 - r.top) / (vh * 0.5)));
      var n = Math.round(p * words.length);
      for (var i = 0; i < words.length; i++) words[i].classList.toggle('on', i < n);
    }
    if (reduced) { words.forEach(function (w) { w.classList.add('on'); }); return; }
    var ticking = false;
    window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(function () { update(); ticking = false; }); } }, { passive: true });
    update();
  })();

  /* ------------------------------------------------------------------ */
  /* 5b. Small screens: jump bar and the short version of each act       */
  /* ------------------------------------------------------------------ */
  (function smallScreens() {
    var main = $('.article-main'), toc = $$('.toc a');
    if (main && toc.length) {
      var bar = el('nav', 'jumpbar'); bar.setAttribute('aria-label', 'Jump to a section');
      toc.forEach(function (a) {
        var link = el('a', 'jump'); link.setAttribute('href', a.getAttribute('href'));
        var num = a.querySelector('small');
        if (num) link.appendChild(el('small', null, num.textContent));
        link.appendChild(document.createTextNode(a.textContent.replace(num ? num.textContent : '', '').trim()));
        bar.appendChild(link);
      });
      main.insertBefore(bar, main.firstChild);
    }
    var mq = window.matchMedia ? window.matchMedia('(max-width: 700px)') : null;
    if (!mq) return;
    $$('.text-block[id]').forEach(function (act) {
      var deep = $$('.deep', act); if (!deep.length) return;
      var btn = el('button', 'more-btn'); btn.type = 'button';
      var lbl = el('span'), chev = el('i', 'chev'); btn.appendChild(lbl); btn.appendChild(chev);
      act.appendChild(btn);
      function apply() {
        var narrow = mq.matches, open = act.classList.contains('is-open');
        deep.forEach(function (d) { if (narrow && !open) d.setAttribute('hidden', ''); else d.removeAttribute('hidden'); });
        btn.hidden = !narrow;
        lbl.textContent = open ? 'Show the short version' : 'Read the full section';
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      btn.addEventListener('click', function () {
        act.classList.toggle('is-open'); apply();
        if (!act.classList.contains('is-open')) act.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
      });
      apply();
      if (mq.addEventListener) mq.addEventListener('change', apply); else if (mq.addListener) mq.addListener(apply);
    });
  })();

  /* ------------------------------------------------------------------ */
  /* 6. Quick-links rail                                                   */
  /* ------------------------------------------------------------------ */
  (function rail() {
    var links = $$('.toc a, .jumpbar a'); if (!links.length) return;
    var map = {}; links.forEach(function (a) { var k = a.getAttribute('href').slice(1); (map[k] = map[k] || []).push(a); });
    if ('IntersectionObserver' in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) current = en.target.id; });
        links.forEach(function (a) { a.classList.toggle('is-active', !!(map[current] && map[current].indexOf(a) !== -1)); if (a.classList.contains('is-active') && a.closest('.jumpbar') && a.scrollIntoView) { try { a.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); } catch (e) {} } });
      }, { rootMargin: '-25% 0px -60% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
    }
    var share = $('#shareBtn'), toast = $('#toast');
    function showToast(msg) { toast.textContent = msg; toast.classList.add('is-on'); clearTimeout(showToast.t); showToast.t = setTimeout(function () { toast.classList.remove('is-on'); }, 2200); }
    if (share) share.addEventListener('click', function () {
      var url = location.href.split('#')[0], title = document.title;
      if (navigator.share) { navigator.share({ title: title, url: url }).catch(function () {}); return; }
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { showToast('Link copied'); }, function () { showToast(url); });
      else showToast(url);
    });
    window.__toast = showToast;
  })();

  /* ------------------------------------------------------------------ */
  /* 7. BibTeX copy, reveal on scroll                                      */
  /* ------------------------------------------------------------------ */
  (function bib() {
    var btn = $('#bibtexCopy'), pre = $('#bibtexText'); if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var txt = pre.textContent;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy BibTeX'; }, 1800); if (window.__toast) window.__toast('BibTeX copied'); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () { selectText(pre); });
      else selectText(pre);
    });
    function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
  })();

  (function reveal() {
    var items = $$('.reveal'); if (!items.length) return;
    if (reduced || !('IntersectionObserver' in window) || /[?&]static\b/.test(location.search)) { items.forEach(function (i) { i.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    items.forEach(function (i) { io.observe(i); });
    /* Failsafe: the reveal is decorative, so nothing may stay hidden if the observer
       never fires (background tab, print, unusual embedding). */
    function revealAll() { items.forEach(function (i) { i.classList.add('in'); }); }
    setTimeout(revealAll, 3500);
    window.addEventListener('beforeprint', revealAll);
  })();
  /* ------------------------------------------------------------------ */
  /* 8. Colab links                                                       */
  /* ------------------------------------------------------------------ */
  /* On GitHub Pages the notebook lives in the same repository, so the Colab
     URL can be derived from the host (USER.github.io) and the first path
     segment (REPO; a user site has none). Anywhere else the link falls back
     to the notebook file itself, which Colab can open via File, Upload. */
  (function colab() {
    var links = $$('a.js-colab').filter(function (a) { return /USER\/REPO/.test(a.getAttribute('href') || ''); });
    if (!links.length) return; /* configure.py has filled in the real link */
    var host = location.hostname, m = /^([a-z0-9-]+)\.github\.io$/i.exec(host);
    var href = null, note = null;
    if (m) {
      var user = m[1], seg = location.pathname.split('/').filter(Boolean)[0];
      var repo = seg && !/\.html?$/i.test(seg) ? seg : user + '.github.io';
      href = 'https://colab.research.google.com/github/' + user + '/' + repo + '/blob/main/colab/sri_playground.ipynb';
    } else {
      href = 'colab/sri_playground.ipynb';
      note = 'This preview is not served from GitHub Pages, so the link downloads the notebook; open it in Colab with File, Upload notebook. Once the site is published on GitHub Pages the same button opens Colab directly.';
    }
    links.forEach(function (a) {
      if (href) a.setAttribute('href', href);
      if (note) { a.setAttribute('title', note); a.removeAttribute('target'); }
    });
  })();
})();
