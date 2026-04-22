import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EpiAgent — LLM selector",
  description: "Filter and rank self-hostable LLMs by capability and compute.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
