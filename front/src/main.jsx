import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import Header from './components/Header'
import { AuthProvider } from './contexts/AuthContext'; 
import './style.css'

ReactDOM
  .createRoot(document.getElementById('app'))
  .render(
    <React.StrictMode>
      <AuthProvider>
        <BrowserRouter>
          <Header />
          <App />
        </BrowserRouter>
      </AuthProvider>
    </React.StrictMode>
  )
