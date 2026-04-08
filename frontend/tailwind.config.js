/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        claudio: {
          bg: '#0a0a0f',
          surface: '#111118',
          card: '#1a1a24',
          border: '#2a2a3a',
          accent: '#00ff88',
          accent2: '#0088ff',
          text: '#e0e0f0',
          muted: '#606080',
        }
      }
    }
  },
  plugins: [],
}
