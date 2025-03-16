// app/layout.tsx
import "../styles/global.css";

export const metadata = {
  title: "Financial Dashboard",
  description: "Comprehensive financial analysis powered by AI agents",
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
