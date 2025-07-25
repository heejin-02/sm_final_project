import { useState, useEffect } from "react";
import { fetchUnreadAlerts } from "../api/noti";

export function useNotifications() {
  const [alerts, setAlerts] = useState([]);
  useEffect(() => {
    fetchUnreadAlerts().then(setAlerts).catch(console.error);
  }, []);
  return alerts;
}
