/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*",  destination: "http://localhost:8000/api/:path*" },
      { source: "/hitl/:path*", destination: "http://localhost:8000/hitl/:path*" },
    ];
  },
};

export default nextConfig;
