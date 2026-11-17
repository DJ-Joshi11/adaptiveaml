/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#08090B", 900: "#0D0E11", 850: "#111215", 800: "#15171B", 750: "#1A1C21", 700: "#202329", 600: "#2A2D34", 500: "#3A3E47" },
        lime: { DEFAULT: "#B6F24A", 300: "#D4FA8C", 400: "#C4F56A", 500: "#B6F24A", 600: "#93D12A", 700: "#5E8A17", 900: "#24350A" },
        mute: "#8D929C",
        soft: "#C9CDD4",
        danger: "#FF5D5D",
        amber: "#F6B44B",
        azure: "#4D8DFF",
        violet: "#A78BFA",
        teal: "#3DD9B3",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      boxShadow: { glow: "0 0 0 1px rgba(182,242,74,.25), 0 8px 30px -8px rgba(182,242,74,.25)", card: "0 1px 0 0 rgba(255,255,255,.04) inset, 0 10px 30px -12px rgba(0,0,0,.6)" },
      keyframes: {
        ticker: { "0%": { transform: "translateX(0)" }, "100%": { transform: "translateX(-50%)" } },
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
        pulseDot: { "0%,100%": { opacity: 1 }, "50%": { opacity: .35 } },
      },
      animation: { ticker: "ticker 60s linear infinite", shimmer: "shimmer 1.4s linear infinite", pulseDot: "pulseDot 1.6s ease-in-out infinite" },
    },
  },
  plugins: [],
};
