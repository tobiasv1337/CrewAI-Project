/* Fit visible Plotly treemap labels to the actual tile dimensions at every zoom.
   Plotly retains the data, hover labels, geometry, and click handlers. */
(() => {
  window.__studyTreemapCleanup?.();
  const svgNS = 'http://www.w3.org/2000/svg';
  const canvas = document.createElement('canvas').getContext('2d');
  const decoder = document.createElement('textarea');
  const cache = new WeakMap();
  const lightPalette = {
    '51,65,85': '#f1f5f9', '57,116,163': '#b9dcf3', '52,127,118': '#b6ded5',
    '110,100,161': '#d4c5ee', '76,137,147': '#bce4e7', '86,109,152': '#c6d5f3',
    '122,103,145': '#e4cdeb', '58,133,133': '#b3e1d8', '98,126,152': '#d1e5f1',
    '70,106,136': '#bcd5ea',
  };
  let frame = 0;
  const fit = () => {
    frame = 0;
    document.querySelectorAll('.treemaplayer').forEach(layer => {
      const portfolio = !!layer.closest('[class*="st-key-portfolio_map_"]');
      const light = document.documentElement.dataset.theme === "light";
      const tiles = Array.from(layer.querySelectorAll('g.slice')).map(group => {
        const surface = group.querySelector('path.surface');
        const label = group.querySelector('text.slicetext');
        return surface && label ? {group, surface, label, box: surface.getBBox()} : null;
      }).filter(Boolean);
      for (const tile of tiles) {
        const {group, surface, label, box} = tile;
        if (portfolio) {
          const current = surface.style.fill;
          const base = current === surface.dataset.smAppliedFill ? surface.dataset.smBaseFill : current;
          surface.dataset.smBaseFill = base;
          const channels = base.match(/[\d.]+/g)?.slice(0, 3).map(Number);
          if (channels?.length === 3) {
            const fill = light ? (lightPalette[channels.join(',')] || '#c8e0ee') : base;
            if (current !== fill) surface.style.fill = fill;
            surface.dataset.smAppliedFill = surface.style.fill;
          }
          const ink = light ? '#243248' : '#ffffff';
          if (label.style.fill !== ink) label.style.fill = ink;
          const stroke = light ? 'rgba(255,255,255,.8)' : 'rgba(255,255,255,.2)';
          if (surface.style.stroke !== stroke) surface.style.stroke = stroke;
          surface.style.strokeOpacity = "1";
        }
        if (box.width < 8 || box.height < 12) continue;
        const raw = label.getAttribute('data-unformatted') || '';
        // Only headers can sit above child tiles; don't draw into their content.
        const childTops = tiles.filter(other => other !== tile && other.box.x >= box.x && other.box.x + other.box.width <= box.x + box.width + .1 && other.box.y > box.y && other.box.y < box.y + box.height).map(other => other.box.y - box.y);
        const height = childTops.length ? Math.min(...childTops) : box.height;
        const fontSize = box.width > 220 && height > 70 ? 16 : box.width > 120 && height > 50 ? 14 : 11;
        const signature = [raw, box.x, box.y, box.width, height, fontSize].join('|');
        const previous = cache.get(label);
        if (previous?.signature === signature && label.textContent === previous.text && label.getAttribute('transform') === previous.transform) continue;
        decoder.innerHTML = raw.replace(/\x3cbr\s*\/?>/gi, ' ').replace(/\x3c[^>]*>/g, '');
        const full = decoder.value.replace(/\s+/g, ' ').trim();
        if (!full) continue;
        canvas.font = `${fontSize}px ${getComputedStyle(label).fontFamily}`;
        const width = Math.max(4, box.width - 10);
        const maxLines = Math.max(1, Math.min(4, Math.floor((height - 8) / (fontSize * 1.25))));
        const lines = [];
        const credit = full.match(/(?:\d+(?:\.\d+)?) LP$/)?.[0] || "";
        let remaining = credit ? full.slice(0, -credit.length).trim() : full;
        const nameLines = Math.max(1, maxLines - (credit && maxLines > 1 ? 1 : 0));
        for (let line = 0; line < nameLines && remaining; line++) {
          if (canvas.measureText(remaining).width <= width) { lines.push(remaining); remaining = ''; break; }
          const last = line === nameLines - 1;
          let end = remaining.length;
          while (end > 0 && canvas.measureText(remaining.slice(0, end) + (last ? '…' : '')).width > width) end--;
          if (!last && end > 2) {
            const space = remaining.lastIndexOf(' ', end);
            if (space > 0) end = space;
          }
          lines.push(remaining.slice(0, end).trimEnd() + (last ? '…' : ''));
          remaining = remaining.slice(Math.max(1, end)).trimStart();
        }
        if (credit && maxLines > 1) lines.push(credit);
        const transform = `translate(${box.x + 5},${box.y + fontSize + 3})`;
        label.replaceChildren();
        lines.forEach((line, index) => {
          const span = document.createElementNS(svgNS, 'tspan');
          span.setAttribute('x', '0'); span.setAttribute('dy', index ? '1.25em' : '0');
          span.textContent = line;
          if (line === credit) span.style.fontWeight = "600";
          label.append(span);
        });
        label.setAttribute('transform', transform);
        label.style.fontSize = `${fontSize}px`;
        label.style.opacity = '1';
        label.style.fillOpacity = '1';
        label.style.pointerEvents = 'none';
        label.parentElement.style.opacity = '1';
        group.setAttribute('aria-label', full);
        cache.set(label, {signature, text: label.textContent, transform});
      }
    });
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(fit); };
  const observer = new MutationObserver(schedule);
  observer.observe(document.body, {subtree:true, childList:true, attributes:true, attributeFilter:['d','transform','data-unformatted']});
  window.addEventListener('resize', schedule);
  window.addEventListener('tu-theme-change', schedule);
  schedule();
  window.__studyTreemapCleanup = () => {observer.disconnect(); cancelAnimationFrame(frame); window.removeEventListener('resize', schedule); window.removeEventListener('tu-theme-change', schedule);};
})();
