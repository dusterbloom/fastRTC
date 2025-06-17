/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone', // This line is key
    
  // Add this block to disable ESLint during builds
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig;