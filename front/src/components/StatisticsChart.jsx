import React from 'react';
import MainBarChart from './MainBarChart';
import InsectPieChart from './InsectPieChart';
import RegionStackedBarChart from './RegionStackedBarChart';

export default function StatisticsChart({ data, period, currentDate }) {
  if (!data) return null;

  return (
    <div className="statistics-charts">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bordered-box p-4 shadow-sm min-w-0">
          <MainBarChart period={period} data={data} currentDate={currentDate} />
        </div>

        <div className="bordered-box p-4 shadow-sm min-w-0">
          <InsectPieChart data={data} />
        </div>

        {(period === 'daily' || period === 'monthly') && (
          <div className="bordered-box lg:col-span-2 p-4 shadow-sm min-w-0">
            <RegionStackedBarChart data={data} period={period} />
          </div>
        )}
      </div>
    </div>
  );
}
