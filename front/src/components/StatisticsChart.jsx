import React from 'react';
import MainBarChart from './MainBarChart';
import InsectPieChart from './InsectPieChart';
import RegionStackedBarChart from './RegionStackedBarChart';

export default function StatisticsChart({ stats, period, currentDate }) {
  if (!stats) return null;

  return (
    <div className='statistics-charts'>
      <div className='grid gap-6 grid-cols-1 lg:grid-cols-2 items-stretch'>
        <div className='bordered-box p-4 shadow-sm'>
          <MainBarChart
            period={period}
            stats={stats}
            currentDate={currentDate}
          />
        </div>

        <div className='bordered-box p-4 shadow-sm'>
          <InsectPieChart stats={stats} />
        </div>

        <div className='bordered-box p-4 shadow-sm col-span-1 lg:col-span-2'>
          <RegionStackedBarChart stats={stats} period={period} />
        </div>
      </div>
    </div>
  );
}
