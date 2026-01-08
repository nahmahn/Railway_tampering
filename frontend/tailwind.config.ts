import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        govt: {
          navy: "#0a2647", // Deep Navy Blue for headers/primary
          blue: "#144272", // Lighter Navy for interactions
          orange: "#ff9933", // Indian Saffron (Accent)
          green: "#138808", // Indian Green (Success/Accent)
          gray: "#f3f4f6", // Light background gray
          border: "#e5e7eb", // Soft border
        },
        alert: {
          red: "#dc2626", // High Priority
          yellow: "#ca8a04", // Warning
          green: "#16a34a", // Safe
        }
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
export default config;
