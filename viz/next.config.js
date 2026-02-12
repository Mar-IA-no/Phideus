/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/Phideus',
  assetPrefix: '/Phideus',
  reactStrictMode: false,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: '/Phideus' },
};

module.exports = nextConfig;
