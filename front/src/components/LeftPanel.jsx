// src/components/LeftPanel.jsx
import React from "react";
import NotiList from "./NotiList";
import DayCheck from "./DayCheck";
import { useNoti } from "../contexts/NotiContext";

export default function LeftPanel() {
  const { isNotiOpen, setIsNotiOpen, setUnreadCount } = useNoti();

  return (
    <div className={`left-panel ${isNotiOpen ? "open" : ""}`}>
      {/* 확인하지 않은 알림 섹션 */}
      <div className="left-panel-section h-[60%] overflow-y-hidden notiList">
        <NotiList />
      </div>

      {/* 오늘의 해충 섹션 */}
      <div className="left-panel-section h-fit scrl-custom dayCheck">
        <DayCheck />
      </div>
    </div>
  );
}
