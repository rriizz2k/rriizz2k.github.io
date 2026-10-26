import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: 'web',
  base: '/',
  cacheDir: '../node_modules/.vite-pages',
  plugins: [react()],
  publicDir: '../public',
  build: {outDir: '../dist-pages', emptyOutDir: true},
});
