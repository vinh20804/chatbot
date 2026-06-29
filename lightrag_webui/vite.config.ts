import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import { fileURLToPath } from 'url'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// ESM __dirname equivalent
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Hardcode webuiPrefix to avoid path alias issues
const webuiPrefix = '/webui/'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, '')
  const backendUrl = env.VITE_BACKEND_URL || 'http://localhost:9621'
  const useProxy = env.VITE_API_PROXY !== 'false'
  
  const defaultEndpoints = [
    '/api', '/documents', '/graphs', '/graph', '/health',
    '/query', '/docs', '/redoc', '/openapi.json', '/login',
    '/auth-status', '/static'
  ]
  const endpoints = env.VITE_API_ENDPOINTS
    ? env.VITE_API_ENDPOINTS.split(',')
    : defaultEndpoints

  const proxy: Record<string, any> = {}
  if (useProxy && backendUrl) {
    endpoints.forEach((endpoint) => {
      const trimmed = endpoint.trim()
      if (trimmed) {
        proxy[trimmed] = {
          target: backendUrl,
          changeOrigin: true,
          ws: true
        }
      }
    })
  }

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      },
      // Force all modules to use the same katex instance
      // This ensures mhchem extension registered in main.tsx is available to rehype-katex
      dedupe: ['katex']
    },
    // base: import.meta.env.VITE_BASE_URL || '/webui/',
    base: webuiPrefix,
    build: {
      outDir: path.resolve(__dirname, '../lightrag/api/webui'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 3800,
      rollupOptions: {
        // Let Vite handle chunking automatically to avoid circular dependency issues
        output: {
          // Ensure consistent chunk naming format
          chunkFileNames: 'assets/[name]-[hash].js',
          // Entry file naming format
          entryFileNames: 'assets/[name]-[hash].js',
          // Asset file naming format
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    server: {
      proxy
    }
  }
})
