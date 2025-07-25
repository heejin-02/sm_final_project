// tailwind.config.js
module.exports = {
  content: [
    './index.html',
		'./src/*.{js,jsx,ts,tsx,css}',
    './src/**/*.{js,jsx,ts,tsx,css}',
  ],
  theme: {
    extend: {
      screens: {
        md: '768px',
        lg: '1024px',
      },
    },
  },
  plugins: [],
};
