// 로딩 스피너
import React from "react";
import { AiOutlineLoading3Quarters } from "react-icons/ai";

export default function Loader({ size = "text-4xl", color = "text-gray-500", message }) {
  return (
    <div className="section flex flex-col items-center justify-center h-full">
      <AiOutlineLoading3Quarters className={`animate-spin ${size} ${color}`} />
      {message && <span className="mt-2 text-sm">{message}</span>}
    </div>
	);}