import React from 'react';

interface StatusBarProps {
    status: string;
    message: string;
}

const StatusBar: React.FC<StatusBarProps> = ({ status, message }) => {
    let statusClass = 'text-bg-secondary';
    if (status === 'running') {
        statusClass = 'text-bg-success';
    } else if (status === 'error') {
        statusClass = 'text-bg-danger';
    } else if (status === 'stopped') {
        statusClass = 'text-bg-warning';
    }

    return (
        <div className="card text-white bg-dark mb-3">
            <div className="card-body">
                <h5 className="card-title">Bot Status</h5>
                <p className="card-text">
                    Current Status: <span className={`badge ${statusClass}`}>{status.toUpperCase()}</span>
                </p>
                <p className="card-text text-muted" style={{ fontSize: '0.9rem' }}>
                    {message}
                </p>
            </div>
        </div>
    );
};

export default StatusBar;
