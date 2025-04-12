import { defineConfig } from 'vite';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import react from '@vitejs/plugin-react';

// Create ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Note: This file was moved to the frontend/ directory as part of the project reorganization
// The relative paths still work because both this file and the referenced directories were moved together
export default defineConfig({
  plugins: [react()],
  base: '/video_sonification/',
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});