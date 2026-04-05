/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          black: "#0a0a0a",
          dark: "#121212",
          gray: "#1e1e1e",
          primary: "#00ff9d", // Neon Green
          secondary: "#7000ff", // Neon Purple
          text: "#e0e0e0"
        }
      },
      fontFamily: {
        mono: ['"Fira Code"', 'monospace'], // Ensure you have a nice mono font or fallback
      }
    },
  },
  plugins: [],
}