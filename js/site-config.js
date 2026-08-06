(function () {
  const ready = fetch('config/portfolio.json')
    .then((response) => {
      if (!response.ok) throw new Error(`Portfolio config: ${response.status}`);
      return response.json();
    })
    .then((config) => {
      window.PORTFOLIO_CONFIG = config;

      document.querySelectorAll('[data-contact-link]').forEach((link) => {
        link.href = `mailto:${config.contactEmail}`;
        if (link.hasAttribute('data-contact-show-email')) {
          const label = link.querySelector('[data-contact-text]') || link;
          label.textContent = config.contactEmail;
        }
      });

      document.querySelectorAll('[data-assistant-name]').forEach((node) => {
        node.textContent = config.assistantName;
      });

      document.querySelectorAll('[data-assistant-description]').forEach((node) => {
        node.textContent = config.assistantDescription;
      });

      document.querySelectorAll('[data-flowentic-link]').forEach((link) => {
        if (config.flowenticSiteEnabled) {
          link.href = config.flowenticSiteUrl;
          link.hidden = false;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.removeAttribute('aria-disabled');
        } else {
          link.hidden = true;
          link.removeAttribute('href');
          link.removeAttribute('target');
          link.setAttribute('aria-disabled', 'true');
        }
      });

      return config;
    })
    .catch((error) => {
      console.warn(error.message);
      return null;
    });

  window.portfolioConfigReady = ready;
})();
