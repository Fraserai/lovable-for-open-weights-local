import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Let a phone or tablet on the same home network use the dev server (http://<this Mac's IP>:3000).
  // Dev only; nothing here is reachable from the internet.
  allowedDevOrigins: ["192.168.*.*", "*.local"],
  // Runs used to open at /runs/<id>; they now live on the Train page's Current Run tab
  async redirects() {
    return [{ source: "/runs/:id", destination: "/?tab=run&run=:id", permanent: false }];
  },
};

export default nextConfig;
