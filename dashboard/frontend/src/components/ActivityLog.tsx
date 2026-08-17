import React, { useEffect, useRef } from 'react';

interface ActivityLogProps {
    logMessages: any[];
}

const ActivityLog: React.FC<ActivityLogProps> = ({ logMessages }) => {
    const logEndRef = useRef<null | HTMLDivElement>(null);

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logMessages]);

    return (
        <div className="card text-white bg-dark mb-3">
            <div className="card-body">
                <h5 className="card-title">Activity Log</h5>
                <pre className="bg-black p-2 rounded" style={{ height: '300px', overflowY: 'scroll', fontSize: '0.8rem' }}>
                    {logMessages.map((msg, index) => (
                        <div key={index} className={`log-${msg.level}`}>
                            <span className="text-muted">[{new Date(msg.timestamp).toLocaleTimeString()}] </span>
                            <span className={`fw-bold text-${msg.level === 'error' ? 'danger' : msg.level === 'opportunity' ? 'success' : 'info'}`}>
                                [{msg.source.toUpperCase()}/{msg.level.toUpperCase()}]
                            </span>
                            : {msg.data.message || JSON.stringify(msg.data)}
                        </div>
                    ))}
                    <div ref={logEndRef} />
                </pre>
            </div>
        </div>
    );
};

export default ActivityLog;
