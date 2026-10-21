import 'katex/dist/katex.min.css';
// The configured theme is resolved to a concrete file by the `@theme` alias in
// vite.config.js. Importing it statically puts it in the main CSS bundle, which
// is linked from <head> and applied before the prerendered HTML is painted.
import '@theme';
import UIkit from 'uikit';
import Icons from 'uikit/dist/js/uikit-icons';

const loadTheme = async (theme) => {
  try {
    switch (theme) {
      case 'dark':
        await import('@/scss/dark-theme.scss');
        break;
      case 'classic':
        await import('@/scss/classic-theme.scss');
        break;
      default:
        await import('@/scss/theme.scss');
    }
  } catch (err) {
    console.error('Failed to load theme:', err);
  }
};

UIkit.use(Icons);

export const toggleTheme = () => {
  const themes = ['light', 'dark', 'classic'];
  const currentTheme = localStorage.getItem('theme') || 'light';
  const nextTheme = themes[(themes.indexOf(currentTheme) + 1) % themes.length];
  loadTheme(nextTheme);
  return nextTheme;
};
