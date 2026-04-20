import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },

  // Dev-only proxy to avoid CORS preflight (OPTIONS) issues when calling LM Studio from the browser.
  // Use base URL `/lmstudio` in the app to route requests through Vite.
  server: {
    proxy: {
      '/lmstudio': {
        target: 'http://127.0.0.1:1234',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/lmstudio/, ''),
      },
    },
  },
})
