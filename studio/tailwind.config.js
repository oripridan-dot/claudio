/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        claudio: {
          bg:      '#080810',
          surface: '#0f0f1a',
          panel:   '#14141f',
          border:  '#1e1e2e',
          accent:  '#f59e0b',   // amber — the human craft colour
          gold:    '#d97706',
          text:    '#e2e8f0',
          muted:   '#64748b',
          danger:  '#ef4444',
          safe:    '#22c55e',
        },
      },
      fontFamily: { studio: ['Inter', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [],
}
