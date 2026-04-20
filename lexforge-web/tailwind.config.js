/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep navy + indigo accent — professional, legal-adjacent
        brand: {
          50:  '#f0f4ff',
          100: '#e0e9ff',
          500: '#5b6eff',
          600: '#4a5aec',
          700: '#3b45c8',
          900: '#1a1f4d',
        },
        risk: {
          low:    '#10b981',  // emerald
          medium: '#f59e0b',  // amber
          high:   '#ef4444',  // red
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
