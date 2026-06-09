/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0f1117",
          card:    "#161b27",
          hover:   "#1c2333",
          border:  "#1e2740",
        },
        brand: {
          DEFAULT: "#00d4aa",
          dim:     "#00d4aa26",
        },
      },
    },
  },
  plugins: [],
}

