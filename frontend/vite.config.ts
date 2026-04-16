import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    port: 5174,
    host: true,
    https: false,
  },
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        if (warning.code === 'TS_ERROR') return;
        warn(warning);
      },
    },
  },
})
