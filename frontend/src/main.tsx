import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

const configuredSiteUrl = import.meta.env.VITE_SITE_URL?.trim().replace(/\/$/, '')
const canonicalUrl = configuredSiteUrl || window.location.origin
let canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]')
if (!canonical) {
  canonical = document.createElement('link')
  canonical.rel = 'canonical'
  document.head.append(canonical)
}
canonical.href = `${canonicalUrl}/`

const openGraphUrl = document.createElement('meta')
openGraphUrl.setAttribute('property', 'og:url')
openGraphUrl.content = `${canonicalUrl}/`
document.head.append(openGraphUrl)

for (const selector of ['meta[property="og:image"]', 'meta[name="twitter:image"]']) {
  const imageMeta = document.querySelector<HTMLMetaElement>(selector)
  if (imageMeta) imageMeta.content = `${canonicalUrl}/social-card.svg`
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
