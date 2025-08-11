// src/api/http.js
import axios from 'axios';

export const API = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' }
});

export const GPT_API = axios.create({
  baseURL: import.meta.env.VITE_FASTAPI_URL,
  headers: { 'Content-Type': 'application/json' }
});
