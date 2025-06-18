# Task Template

## 1. Task & Context
**Task:** Configure production build integration between frontend and backend for containerized deployment
**Scope:** backend/start_clean.py static file serving, frontend/react-vite/next.config.ts
**Branch:** feature/docker-production-build

## 2. Quick Plan
**Approach:** Update backend to properly serve frontend static files in production and configure Next.js for container deployment
**Complexity:** 2-Moderate
**Uncertainty:** 2-Medium
**Unknowns:** Next.js static export compatibility with FastAPI serving, optimal asset path configuration
**Human Input Needed:** No

## 3. Implementation
```python
# backend/start_clean.py - line ~195
# Update static file serving for production containers
_frontend_dist = Path(__file__).parent.parent / "frontend" / "react-vite" / "dist"
_frontend_out = Path(__file__).parent.parent / "frontend" / "react-vite" / "out"

# Try Next.js static export first, then fallback to dist
if _frontend_out.exists():
    app.mount("/", StaticFiles(directory=_frontend_out, html=True), name="spa")
    logger.info("🌐 Serving frontend from Next.js static export")
elif _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")
    logger.info("🌐 Serving frontend from dist directory")
else:
    logger.warning("⚠️ No frontend build found - API only mode")
```

```typescript
// frontend/react-vite/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable static export for containerized deployment
  output: process.env.NODE_ENV === 'production' ? 'export' : undefined,
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Disable server features that don't work with static export
  typescript: {
    ignoreBuildErrors: false,
  }
};

export default nextConfig;
```

```json
// frontend/react-vite/package.json - update scripts section
"scripts": {
  "dev": "next dev --turbopack",
  "build": "next build",
  "build:static": "next build && next export",
  "start": "next start",
  "lint": "next lint"
}
```

## 4. Check & Commit
**Changes Made:**
- Updated backend static file serving to support Next.js static export
- Configured Next.js for static export in production builds
- Added fallback logic for different frontend build outputs
- Added build:static script for containerized deployment

**Commit Message:** feat: configure production build integration for containerized deployment

**Status:** Complete
