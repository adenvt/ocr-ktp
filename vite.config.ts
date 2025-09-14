import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
  plugins      : [tailwindcss(), nodePolyfills()],
  assetsInclude: ['**/*.onnx', '**/*.ort'],
  optimizeDeps : { exclude: ['onnxruntime-web'] },
})
