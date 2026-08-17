import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import StatusBar from './components/StatusBar';
import PriceChart from './components/PriceChart';
import ActivityLog from './components/ActivityLog';
import './App.css';

const SOCKET_SERVER_URL = "http://localhost:3001";
const MAX_PRICE_POINTS = 50; // Keep last 50 data points for the chart
const MAX_LOG_MESSAGES = 100; // Keep last 100 log messages

function App() {
    const [status, setStatus] = useState('connecting');
    const [statusMessage, setStatusMessage] = useState('Connecting to dashboard backend...');
    const [priceHistory, setPriceHistory] = useState<any[]>([]);
    const [logMessages, setLogMessages] = useState<any[]>([]);
    const [currentPair, setCurrentPair] = useState('N/A');

    useEffect(() => {
        const socket = io(SOCKET_SERVER_URL);

        socket.on('connect', () => {
            setStatus('running');
            setStatusMessage('Connected. Waiting for bot data...');
        });

        socket.on('disconnect', () => {
            setStatus('stopped');
            setStatusMessage('Disconnected from backend. Is the backend server running?');
        });

        socket.on('bot_update', (msg) => {
            const timestamp = new Date();
            
            if (msg.type === 'opportunity_found') {
                const newDataPoint = {
                    timestamp: timestamp,
                    cexPrice: msg.data.cex_price,
                    dexPrice: msg.data.dex_price,
                };
                setPriceHistory(prev => [...prev.slice(-MAX_PRICE_POINTS + 1), newDataPoint]);
                setCurrentPair(msg.data.cex_symbol || 'N/A');
            }

            const newLog = {
                timestamp: timestamp,
                source: msg.source,
                level: msg.type === 'opportunity_found' ? 'opportunity' : msg.type,
                data: msg.data
            };
            setLogMessages(prev => [...prev.slice(-MAX_LOG_MESSAGES + 1), newLog]);
        });

        return () => {
            socket.disconnect();
        };
    }, []);

    return (
        <div className="App container-fluid py-3">
            <header className="App-header mb-3">
                <h2>DEX-CEX Arbitrage Bot Dashboard</h2>
            </header>
            <div className="row">
                <div className="col-md-4">
                    <StatusBar status={status} message={statusMessage} />
                    <ActivityLog logMessages={logMessages} />
                </div>
                <div className="col-md-8">
                    <PriceChart priceHistory={priceHistory} pair={currentPair} />
                </div>
            </div>
        </div>
    );
}

export default App;