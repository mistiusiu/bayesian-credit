import { fileURLToPath, URL } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

// `/api` is proxied to the Django backend in dev and preview. Override the target
// with VITE_API_PROXY_TARGET (e.g. when the API runs in docker on another port).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8010';
  const proxy = { '/api': { target, changeOrigin: true } };

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: { port: 5173, proxy },
    preview: { port: 4173, proxy },
    build: {
      target: 'es2023',
      sourcemap: true,
      chunkSizeWarningLimit: 1200,
    },
  };
});
