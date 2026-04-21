import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LexForge AI — Fine-tuning Console",
  description:
    "Submit a QLoRA SFT job to Modal. Stream live logs. Merge and push to HuggingFace.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
