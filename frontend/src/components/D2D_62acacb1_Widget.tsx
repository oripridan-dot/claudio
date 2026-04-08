import React, { useState, useEffect } from 'react';
import { useEventListener } from '@react-hookz/web';

interface D2DWidgetProps {
    onConnect: () => void;
    onDisconnect: () => void;
    isConnected: boolean;
}

const D2DWidget: React.FC<D2DWidgetProps> = ({ onConnect, onDisconnect, isConnected }) => {
    const [status, setStatus] = useState<string>("Disconnected");

    useEffect(() => {
        setStatus(isConnected ? "Connected" : "Disconnected");
    }, [isConnected]);

    const handleConnectClick = () => {
        onConnect();
    };

    const handleDisconnectClick = () => {
        onDisconnect();
    };


    return (
        <div className="bg-gray-800 p-4 rounded-md shadow-md">
            <h2 className="text-xl font-bold text-white mb-2">Claudio Core Connection</h2>
            <p className="text-gray-300">Status: {status}</p>

            {isConnected ? (
                <button
                    onClick={handleDisconnectClick}
                    className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-2"
                >
                    Disconnect
                </button>
            ) : (
                <button
                    onClick={handleConnectClick}
                    className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mt-2"
                >
                    Connect
                </button>
            )}
        </div>
    );
};

export default D2DWidget;
