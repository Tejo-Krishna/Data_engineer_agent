"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export default function AppHeader() {
  const path = usePathname();

  const nav = [
    { href: "/wizard", label: "New Run" },
    { href: "/runs",   label: "History" },
  ];

  return (
    <header className="border-b bg-white sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-8">
        <Link href="/wizard" className="font-bold text-lg tracking-tight">
          Data Agent
        </Link>
        <nav className="flex gap-4">
          {nav.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={cn(
                "text-sm font-medium transition-colors",
                path.startsWith(n.href)
                  ? "text-black"
                  : "text-gray-500 hover:text-black"
              )}
            >
              {n.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
