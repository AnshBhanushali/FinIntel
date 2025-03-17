/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
    turbo: false, // Disable Turbopack
  },
};

module.exports = nextConfig;
