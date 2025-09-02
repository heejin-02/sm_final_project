import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import logo from '/images/logo-horizon.svg';
import { LuLogOut, LuMenu, LuMinus, LuPlus } from 'react-icons/lu';

export default function Header() {
    const { user, logout } = useAuth(); // logout 함수도 가져옴
    const navigate = useNavigate();
    const [zoomLevel, setZoomLevel] = useState('100%');
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    // 스크롤 이벤트
    useEffect(() => {
        const handleScroll = () => {
            const bwLeft = window.scrollX;
            document.querySelector('.header-in').style.transform = `translateX(-${bwLeft}px)`;
        };
        window.addEventListener('scroll', handleScroll);

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    // 컴포넌트 마운트 시 저장된 확대/축소 레벨 불러오기
    useEffect(() => {
        const savedZoom = localStorage.getItem('zoomLevel') || '100%';
        setZoomLevel(savedZoom);
        applyZoom(savedZoom);
    }, []);

    // 확대/축소 적용 함수
    const applyZoom = (zoom) => {
        const html = document.documentElement;
        // 기존 data-zoom 속성 제거
        html.removeAttribute('data-zoom');
        // 새로운 확대/축소 레벨 적용
        if (zoom !== '100%') {
            html.setAttribute('data-zoom', zoom);
        }
    };

    // 확대 함수
    const handleZoomIn = () => {
        let newZoom;
        switch (zoomLevel) {
            case '90%':
                newZoom = '100%';
                break;
            case '100%':
                newZoom = '110%';
                break;
            case '110%':
                newZoom = '120%';
                break;
            default:
                newZoom = '120%'; // 최대 120%
        }
        setZoomLevel(newZoom);
        applyZoom(newZoom);
        localStorage.setItem('zoomLevel', newZoom);
    };

    // 축소 함수
    const handleZoomOut = () => {
        let newZoom;
        switch (zoomLevel) {
            case '120%':
                newZoom = '110%';
                break;
            case '110%':
                newZoom = '100%';
                break;
            case '100%':
                newZoom = '90%';
                break;
            default:
                newZoom = '90%'; // 최소 90%
        }
        setZoomLevel(newZoom);
        applyZoom(newZoom);
        localStorage.setItem('zoomLevel', newZoom);
    };

    const handleLogout = () => {
        logout(); // AuthContext의 logout 함수 호출
        navigate('/', { replace: true }); // home.jsx로 이동
    };

    return (
        <header className={`header ${isMenuOpen ? 'open' : ''}`}>
            <div className='header-in'>
                {/* 왼쪽 영역 - 로고 */}
                <div className='header-left'>
                    <div className='logo-area cursor-pointer' onClick={() => navigate('/')}>
                        <img src={logo} alt='로고' className='logo pc' />
                        <div className='mobile'>
                            <img src='/images/m_logo_symbol.png' alt='로고' className='logo' />
                        </div>
                    </div>
                </div>

                {/* 가운데 영역 - 농장명 */}
                <div className='header-center'>
                    {user?.userName && user.role !== 'admin' && user.selectedFarm?.farmName && (
                        <div
                            className='header-text cursor-pointer transition-colors'
                            onClick={() => {
                                // 현재 페이지가 MainFarm이면 새로고침, 아니면 이동
                                if (window.location.pathname === `/mainFarm/${user.selectedFarm.farmIdx}`) {
                                    window.location.reload();
                                } else {
                                    navigate(`/mainFarm/${user.selectedFarm.farmIdx}`);
                                }
                            }}
                        >
                            {user.selectedFarm.farmName} <span className='font-normal text-black hover:text-black'>관리중</span>
                        </div>
                    )}
                </div>

                {/* 오른쪽 영역 - 사용자 정보 & 날씨 */}
                <div className='header-right'>
                    {user?.role === 'admin' ? (
                        <div className='flex items-center justify-end gap-2'>
                            <span>관리자 모드</span>
                            <LuLogOut size={24} onClick={handleLogout} className='cursor-pointer' />
                        </div>
                    ) : (
                        <div className='user'>
                            <div className='pc'>
                                <div className='view-control'>
                                    <button className='btn' onClick={handleZoomOut} disabled={zoomLevel === '90%'}>
                                        작게
                                    </button>
                                    <span>글씨 크기</span>
                                    <button className='btn' onClick={handleZoomIn} disabled={zoomLevel === '120%'}>
                                        크게
                                    </button>
                                </div>

                                {user?.userName && (
                                    <div className='flex items-center gap-5'>
                                        <span className='header-text'>{user.userName}님 환영합니다</span>
                                        <button className='color-80 cursor-pointer' onClick={handleLogout}>
                                            로그아웃
                                        </button>
                                    </div>
                                )}
                            </div>
                            <div className='mobile'>
                                <div className='m-menu cursor-pointer' onClick={() => setIsMenuOpen((prev) => !prev)}>
                                    <LuMenu size={24} />
                                    <div>메뉴</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <div className='m-nav'>
                <div className='nav-cont flex flex-col space-y-4 scrl-custom'>
                    <div className='view-control justify-end gap-4'>
                        <div>화면</div>
                        <div className='flex gap-1'>
                            <button className='btn' onClick={handleZoomOut} disabled={zoomLevel === '90%'}>
                                <LuMinus size={20} />
                                작게 보기
                            </button>
                            <button className='btn' onClick={handleZoomIn} disabled={zoomLevel === '120%'}>
                                <LuPlus size={20} />
                                크게 보기
                            </button>
                        </div>
                    </div>
                    {user?.userName && (
                        <div className='nav-user'>
                            <div className='header-text'>{user.userName}님 환영합니다</div>
                            <button className='color-80 cursor-pointer' onClick={handleLogout}>
                                로그아웃
                            </button>
                        </div>
                    )}
                    <div className='nav-link'>
                        <div>다른 농장 선택</div>
                        <br />
                        <div>오늘의 알림</div>
                        <div></div>
                    </div>
                </div>
            </div>
        </header>
    );
}
