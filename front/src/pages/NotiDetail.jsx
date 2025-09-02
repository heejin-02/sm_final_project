// src/pages/NotiDetail.jsx
import { useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import LeftPanel from '../components/LeftPanel';
import { useAlertDetail } from '../hooks/useAlerts';
import { useRegions } from '../hooks/useRegions';
import { useAuth } from '../contexts/AuthContext';
import { useDataCache } from '../contexts/DataCacheContext';
import BaseFarmMap from '../components/NotiFarmMap';
import DetectionFeedback from '../components/DetectionFeedback';
import Loader from '../components/Loader';

export default function NotiDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const anlsIdx = parseInt(id);
    const { user } = useAuth();
    const farmIdx = user?.selectedFarm?.farmIdx;
    const { findGhIdxByName } = useDataCache();

    const { alertDetail, loading: alertLoading, error } = useAlertDetail(anlsIdx);
    const { regions, loading: regionsLoading } = useRegions();

    // ghIdx 찾기 로직 (useMemo로 최적화 및 렌더링 중 상태 업데이트 방지)
    const targetGhIdx = useMemo(() => {
        if (!alertDetail) return null;

        // 1. 직접적인 ghIdx 확인
        const directGhIdx =
            alertDetail.ghIdx || alertDetail.greenhouseInfo?.ghIdx || alertDetail.greenhouse?.ghIdx || alertDetail.anlsGhIdx;

        if (directGhIdx) {
            return directGhIdx;
        }

        // 2. 캐시된 구역 데이터에서 ghName으로 찾기 (더 정확함)
        const ghName = alertDetail.greenhouseInfo?.ghName;
        if (ghName && farmIdx) {
            const cachedGhIdx = findGhIdxByName(farmIdx, ghName);
            if (cachedGhIdx) {
                return cachedGhIdx;
            }
        }

        // 3. fallback: regions에서 찾기
        if (ghName && regions.length > 0) {
            const foundRegion = regions.find((r) => r.name === ghName);
            if (foundRegion) {
                return foundRegion.id;
            }
        }

        return null;
    }, [alertDetail, farmIdx, findGhIdxByName, regions]);

    // 페이지 진입 시 body 스크롤 막기
    useEffect(() => {
        document.body.style.overflow = 'hidden';
        document.documentElement.style.overflow = 'hidden';

        return () => {
            document.body.style.overflow = 'unset';
            document.documentElement.style.overflow = 'unset';
        };
    }, []);

    // 피드백 제출 핸들러 → axios 대신 alert
    const handleFeedbackSubmit = (feedback) => {
        // 나중에 실제 API 붙일 때 여기에 axios.post 넣으면 됩니다.
    };

    const handleMarkAsRead = () => {
        // 읽음 처리 로직(없으면 빈 함수)
    };

    // 로딩 중
    if (alertLoading || regionsLoading) {
        return (
            <div className='noti-detail-container'>
                <div className='left-panel-wrapper'>
                    <LeftPanel />
                </div>
                <div className='right-section flex items-center justify-center'>
                    <Loader message='알림 상세 정보를 불러오는 중...' />
                </div>
            </div>
        );
    }

    // 에러 또는 알림을 찾지 못한 경우
    if (error || !alertDetail) {
        return (
            <div className='noti-detail-container'>
                <div className='left-panel-wrapper'>
                    <LeftPanel />
                </div>
                <div className='right-section'>
                    <div className='text-center p-8'>
                        <h1 className='text-2xl font-bold mb-4'>알림을 찾을 수 없습니다</h1>
                        <p className='text-gray-600 mb-4'>요청하신 알림이 존재하지 않거나 삭제되었습니다.</p>
                        {/* <button
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button> */}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className='noti-detail-container'>
            {/* 왼쪽 패널 - 모바일에서는 숨김 */}
            <div className='left-panel-wrapper'>
                <LeftPanel />
            </div>

            {/* 오른쪽 컨텐츠 영역 */}
            <div className='right-section space-y-6'>
                {/* 헤더 */}
                <div className='r-sec-top'>
                    <div>
                        <p className='tit'>오늘의 알림</p>
                        <p className='desc'>탐지된 해충에 대한 상세 정보입니다.</p>
                    </div>
                    {/* <button 
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button> */}
                </div>

                {/* 탐지 영상 */}
                <div className='detection-content-wrapper'>
                    <div className='bordered-box detection-box'>
                        {/* <h3 className="tit-2 text-center">탐지 구역</h3> */}
                        <div className='text-center mb-3'>
                            <span className='text-gray-600 text-lg'>
                                <span className='font-semibold text-black'>
                                    {alertDetail.greenhouseInfo?.ghName || `${alertDetail.greenhouseInfo?.ghIdx}번 구역`}
                                </span>{' '}
                                에서&nbsp;
                                <span className='font-semibold text-black'>{alertDetail.greenhouseInfo?.insectName}</span> 탐지됨&nbsp;
                                <span className='text-base'>(신뢰도 {alertDetail.greenhouseInfo?.anlsAcc}%)</span>
                            </span>
                            <div>{alertDetail.greenhouseInfo?.createdAt}</div>
                        </div>
                        <BaseFarmMap
                            highlightRegion={alertDetail.greenhouseInfo?.ghName}
                            highlightGhIdx={targetGhIdx}
                            regions={regions}
                            loading={regionsLoading}
                            gap={0}
                            useApiData={false}
                        />
                    </div>
                    <div className='bordered-box detection-box'>
                        <h3 className='tit-2 text-center'>탐지 영상</h3>
                        <div className='video_wrap'>
                            {alertDetail.imageList?.[0]?.imgUrl ? (
                                <>
                                    <video
                                        src={alertDetail.imageList[0].imgUrl}
                                        controls
                                        muted
                                        autoPlay
                                        playsInline
                                        crossOrigin='anonymous'
                                        onError={(e) => {
                                            console.error('Video load error:', e);
                                            console.error('Failed video URL:', alertDetail.imageList[0].imgUrl);
                                            console.error('Video error code:', e.target.error?.code);
                                            console.error('Video error message:', e.target.error?.message);

                                            // 에러 코드별 메시지
                                            let errorMsg = '동영상을 로드할 수 없습니다.';
                                            if (e.target.error?.code === 1) errorMsg = '비디오 로딩이 중단되었습니다.';
                                            if (e.target.error?.code === 2) errorMsg = '네트워크 오류가 발생했습니다.';
                                            if (e.target.error?.code === 3) errorMsg = '비디오 디코딩 오류가 발생했습니다.';
                                            if (e.target.error?.code === 4) errorMsg = '지원되지 않는 비디오 형식입니다.';

                                            // 대체 컨텐츠 표시
                                            const fallbackDiv = document.createElement('div');
                                            fallbackDiv.className =
                                                'flex flex-col items-center justify-center h-64 bg-gray-100 text-gray-600 p-4';
                                            fallbackDiv.innerHTML = `
                          <p class="mb-2">🎥 ${errorMsg}</p>
                          <p class="text-sm mb-4">URL: ${alertDetail.imageList[0].imgUrl}</p>
                          <a href="${alertDetail.imageList[0].imgUrl}" target="_blank" class="text-blue-500 hover:underline">
                            브라우저에서 직접 열기
                          </a>
                        `;
                                            e.target.parentNode.replaceChild(fallbackDiv, e.target);
                                        }}
                                        onLoadedData={(e) => {
                                            console.log('✅ Video loaded successfully');
                                            console.log('Video duration:', e.target.duration);
                                            console.log('Video dimensions:', e.target.videoWidth, 'x', e.target.videoHeight);
                                        }}
                                    />
                                </>
                            ) : (
                                <div className='flex items-center justify-center h-64 bg-gray-100 text-gray-500'>동영상이 없습니다.</div>
                            )}
                        </div>
                    </div>
                </div>

                {/* AI 탐지 결과 피드백 */}
                <DetectionFeedback
                    anlsIdx={anlsIdx}
                    alertDetail={alertDetail}
                    onFeedbackSubmit={handleFeedbackSubmit}
                    onMarkAsRead={handleMarkAsRead}
                />
            </div>
        </div>
    );
}
