import { defineConfig } from "vite";

export default defineConfig({
  preview: {
    allowedHosts: [process.env.VITE_APP_HOST]
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        secure: false
      }
    }
  }
});
