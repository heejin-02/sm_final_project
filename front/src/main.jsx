import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import Header from './components/Header';
import { AuthProvider } from './contexts/AuthContext';
import { NotiProvider } from './contexts/NotiContext';
import './style.css';

// StrictMode는 개발 환경에서만 활성화 (중복 실행 방지)
const isDevelopment = import.meta.env.DEV;

const AppWrapper = ({ children }) => {
    if (isDevelopment) {
        // 개발 환경에서는 StrictMode 비활성화 (중복 API 호출 방지)
        return children;
    }
    return <React.StrictMode>{children}</React.StrictMode>;
};

ReactDOM.createRoot(document.getElementById('app')).render(
    <AppWrapper>
        <AuthProvider>
            <BrowserRouter>
                <NotiProvider>
                    <Header />
                    <App />
                </NotiProvider>
            </BrowserRouter>
        </AuthProvider>
    </AppWrapper>,
);
