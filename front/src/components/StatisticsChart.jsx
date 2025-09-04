import React from 'react';
import MainBarChart from './MainBarChart';
import InsectPieChart from './InsectPieChart';
import RegionStackedBarChart from './RegionStackedBarChart';

export default function StatisticsChart({ stats, period, currentDate }) {
  if (!stats) return null;

  return (
    <div className='statistics-charts'>
      <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
        <div className='bordered-box p-4 shadow-sm min-w-0'>
          <MainBarChart
            period={period}
            stats={stats}
            currentDate={currentDate}
          />
        </div>

        <div className='bordered-box p-4 shadow-sm min-w-0'>
          <InsectPieChart stats={stats} />
        </div>

        <div className='bordered-box lg:col-span-2 p-4 shadow-sm min-w-0'>
          <RegionStackedBarChart stats={stats} period={period} />
        </div>
      </div>
    </div>
  );
}
