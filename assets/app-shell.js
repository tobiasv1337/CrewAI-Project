/* Runs in the Streamlit document, not a disposable component iframe. */
(() => {
  window.__studyShellCleanup?.();
  const doc = document;
  const controller = new AbortController();
  const signal = controller.signal;
  let frame = 0;
  let lastTheme = '';
  let lastRoute = null;
  const syncTheme = () => {
    frame = 0;
    const params = new URLSearchParams(location.search);
    const route = [params.get('page'), params.get('module_id')].join(':');
    if (lastRoute !== null && route !== lastRoute) {
      const main = doc.querySelector('[data-testid="stMain"]');
      if (main) main.scrollTop = 0;
    }
    lastRoute = route;
    // stHeader retains Streamlit's native background, including an explicit
    // Light/Dark choice. Custom app surfaces must never be the input here.
    const header = doc.querySelector('[data-testid="stHeader"]');
    const rgb = header && getComputedStyle(header).backgroundColor.match(/[\d.]+/g);
    const theme = rgb && rgb.length >= 3 && (rgb.length < 4 || +rgb[3] > 0)
      ? (.2126 * +rgb[0] + .7152 * +rgb[1] + .0722 * +rgb[2] < 140 ? 'dark' : 'light')
      : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    for (const el of [doc.documentElement, doc.body, doc.querySelector('.stApp')]) {
      if (el && el.dataset.theme !== theme) el.dataset.theme = theme;
    }
    if (lastTheme !== theme) {
      lastTheme = theme;
      window.dispatchEvent(new CustomEvent('tu-theme-change', {detail: {theme}}));
    }
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(syncTheme); };
  const observer = new MutationObserver(schedule);
  observer.observe(doc.body, {subtree: true, childList: true, attributes: true, attributeFilter: ['class']});
  observer.observe(doc.head, {subtree: true, childList: true});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', schedule, {signal});
  syncTheme();

  doc.getElementById('nm-hamburger-bar')?.remove();
  doc.getElementById('sm-mobile-nav')?.remove();
  const bar = doc.createElement('div');
  bar.id = 'sm-mobile-nav';
  const menuButton = doc.createElement('button');
  menuButton.type = 'button';
  menuButton.setAttribute('aria-label', 'Open navigation');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-controls', 'sm-sidebar');
  menuButton.textContent = '☰';
  const brand = doc.createElement('span');
  brand.textContent = 'Study Manager';
  bar.append(menuButton, brand);
  doc.body.append(bar);
  let backdrop = doc.getElementById('sm-nav-backdrop');
  if (!backdrop) { backdrop = doc.createElement('button'); backdrop.id = 'sm-nav-backdrop'; backdrop.setAttribute('aria-label', 'Close navigation'); doc.body.append(backdrop); }
  const toggle = bar.querySelector('button');
  let desktopCollapsed = window.__smDesktopCollapsed === true;
  const setOpen = (open) => {
    doc.documentElement.classList.toggle('sm-nav-open', open);
    const mobile = innerWidth <= 900;
    const expanded = mobile ? open : !desktopCollapsed;
    doc.documentElement.classList.toggle('sm-sidebar-collapsed', desktopCollapsed);
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', mobile ? (open ? 'Close navigation' : 'Open navigation') : (expanded ? 'Collapse sidebar' : 'Expand sidebar'));
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    if (sidebar) { sidebar.id = 'sm-sidebar'; sidebar.inert = mobile ? !open : desktopCollapsed; }
    if (!open && sidebar?.contains(doc.activeElement)) toggle.focus();
  };
  toggle.addEventListener('click', () => {
    if (innerWidth <= 900) setOpen(!doc.documentElement.classList.contains('sm-nav-open'));
    else {
      desktopCollapsed = !desktopCollapsed;
      window.__smDesktopCollapsed = desktopCollapsed;
      setOpen(false);
    }
  }, {signal});
  backdrop.addEventListener('click', () => setOpen(false), {signal});
  doc.addEventListener('keydown', e => { if (e.key === 'Escape') setOpen(false); }, {signal});
  doc.addEventListener('change', e => { if (e.target.closest('[data-testid="stSidebar"] [role="radiogroup"]')) setOpen(false); }, {signal});
  window.addEventListener('resize', () => setOpen(false), {signal});
  setOpen(false);
  window.__studyShellCleanup = () => { controller.abort(); observer.disconnect(); cancelAnimationFrame(frame); };
})();
