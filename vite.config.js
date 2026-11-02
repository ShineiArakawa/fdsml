import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import yaml from '@rollup/plugin-yaml';
import path from 'path';
import fs from 'fs';
import jsyaml from 'js-yaml';

// Resolve the theme at build time so its stylesheet can be imported statically.
// A static import lands in the main CSS bundle linked from <head>, so the
// prerendered HTML is styled on first paint (no flash of unstyled content).
const template = jsyaml.load(
  fs.readFileSync(path.resolve(__dirname, 'template.yaml'), 'utf8')
);
const themeFiles = {
  dark: 'dark-theme.scss',
  classic: 'classic-theme.scss',
};
const themeFile = themeFiles[template.theme] || 'theme.scss';

// https://vite.dev/config/
export default defineConfig({
  base: './',
  resolve: {
    alias: [
      {
        find: /^@theme$/,
        replacement: path.resolve(__dirname, 'src/scss', themeFile),
      },
      { find: '@', replacement: path.resolve(__dirname, 'src') },
    ],
  },
  plugins: [react(), yaml()],
  build: {
    outDir: 'build',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
      },
    },
    target: 'es2015',
    cssMinify: true,
  },
  server: {
    host: '0.0.0.0',
    port: 8080,
  },
  css: {
    preprocessorOptions: {
      scss: {
        silenceDeprecations: ['import'],
        quietDeps: true,
      },
    },
  },
});
