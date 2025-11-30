import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Set base path for deploying to a subdirectory
  // Change this to match your deployment path (e.g., '/weemap-scanner/')
  base: '/demos/weemap-scanner/',
})
