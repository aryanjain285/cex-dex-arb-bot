import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale
);

interface PriceChartProps {
    priceHistory: any[];
    pair: string;
}

const PriceChart: React.FC<PriceChartProps> = ({ priceHistory, pair }) => {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top' as const,
                labels: {
                    color: '#fff'
                }
            },
            title: {
                display: true,
                text: `${pair} Price Feed`,
                color: '#fff'
            },
        },
        scales: {
            x: {
                type: 'time' as const,
                time: {
                    unit: 'second' as const,
                    displayFormats: { 
                        second: 'HH:mm:ss'
                    }
                },
                ticks: {
                    color: '#fff'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            y: {
                ticks: {
                    color: '#fff'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        }
    };

    const data = {
        labels: priceHistory.map(p => p.timestamp),
        datasets: [
            {
                label: 'CEX Price',
                data: priceHistory.map(p => p.cexPrice),
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                tension: 0.1
            },
            {
                label: 'DEX Price',
                data: priceHistory.map(p => p.dexPrice),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                tension: 0.1
            },
        ],
    };

    return (
        <div className="card text-white bg-dark mb-3" style={{ height: '400px' }}>
            <div className="card-body">
                <Line options={options} data={data} />
            </div>
        </div>
    );
};

export default PriceChart;
