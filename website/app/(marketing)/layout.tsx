import { Nav } from "@/components/marketing/Nav";
import { Footer } from "@/components/marketing/Footer";
import { TickerTape } from "@/components/marketing/TickerTape";

export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col">
      <TickerTape />
      <Nav />
      <main className="flex-1">{children}</main>
      <Footer />
    </div>
  );
}
