import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Identity Resolution Demo - Probabilistic Cross-Device Attribution",
  description: "Interactive demonstration of probabilistic identity resolution and multi-touch attribution modeling. Visualize how users interact across devices and how credit is assigned to touchpoints.",
  keywords: ["identity resolution", "cross-device attribution", "multi-touch attribution", "probabilistic matching", "marketing analytics"],
  authors: [{ name: "Probabilistic Identity Resolution" }],
  openGraph: {
    title: "Identity Resolution Demo",
    description: "Visualize cross-device user journeys and attribution models",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
