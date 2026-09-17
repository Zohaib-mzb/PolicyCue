import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { loadEnv, type Plugin } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const publicSiteUrl = loadEnv(mode, process.cwd(), '').VITE_SITE_URL?.trim().replace(/\/$/, '')
  const sitemapPlugin: Plugin = {
    name: 'policycue-sitemap',
    generateBundle() {
      if (!publicSiteUrl || !/^https:\/\//i.test(publicSiteUrl)) return
      this.emitFile({
        type: 'asset',
        fileName: 'sitemap.xml',
        source: `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>${publicSiteUrl}/</loc></url></urlset>\n`,
      })
    },
  }
  return {
    plugins: [react(), sitemapPlugin],
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
    },
  }
})
