import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './style.css'
import Header from './components/Header'

ReactDOM
  .createRoot(document.getElementById('app'))
  .render(
    <BrowserRouter>
      <Header/>
      <App />
    </BrowserRouter>
  )
