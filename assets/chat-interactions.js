/* Keep inspected A2A messages open while other agents stream new events. */
(() => {
  window.__studyChatCleanup?.();
  const expanded = window.__studyChatExpanded ??= new Map();
  const restoring = new WeakSet();
  let frame = 0;
  const restore = () => {
    frame = 0;
    document.querySelectorAll('details[data-dialogue-key]').forEach(details => {
      const saved = expanded.get(details.dataset.dialogueKey);
      if (saved !== undefined && details.open !== saved) {
        restoring.add(details);
        details.open = saved;
      }
    });
  };
  const remember = event => {
    const details = event.target;
    if (!(details instanceof HTMLDetailsElement) || !details.dataset.dialogueKey || !details.isConnected) return;
    if (restoring.has(details)) { restoring.delete(details); return; }
    expanded.set(details.dataset.dialogueKey, details.open);
    if (expanded.size > 500) expanded.delete(expanded.keys().next().value);
  };
  const observer = new MutationObserver(() => { if (!frame) frame = requestAnimationFrame(restore); });
  document.addEventListener('toggle', remember, true);
  observer.observe(document.body, {childList: true, subtree: true});
  restore();
  window.__studyChatCleanup = () => {
    observer.disconnect();
    document.removeEventListener('toggle', remember, true);
    cancelAnimationFrame(frame);
  };
})();
