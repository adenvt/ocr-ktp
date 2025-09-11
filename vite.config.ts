import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins      : [tailwindcss()],
  assetsInclude: ['**/*.onnx'],
  optimizeDeps : { exclude: ['onnxruntime-web'] },
})
