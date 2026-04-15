import { useEffect, useRef, useState } from 'react';

/**
 * WebSocket hook — connects to Claudio Intelligence Server.
 * Handles reconnection and message routing.
 */
export function useClaudioSocket(url = `${(import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/^http/, 'ws')}/ws/session`) {

  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const listenersRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());

  useEffect(() => {
    let ws: WebSocket;
    let retryTimeout: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setLastMessage(msg);
          const handlers = listenersRef.current.get(msg.type);
          if (handlers) {
            handlers.forEach((fn) => fn(msg.data ?? msg));
          }
        } catch { /* ignore non-JSON */ }
      };

      ws.onclose = () => {
        setConnected(false);
        retryTimeout = setTimeout(connect, 2000);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      clearTimeout(retryTimeout);
      ws?.close();
    };
  }, [url]);

  const send = (type: string, data: Record<string, any> = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, ...data }));
    }
  };

  const on = (type: string, handler: (data: any) => void) => {
    if (!listenersRef.current.has(type)) {
      listenersRef.current.set(type, new Set());
    }
    listenersRef.current.get(type)!.add(handler);
    return () => { listenersRef.current.get(type)?.delete(handler); };
  };

  return { connected, send, on, lastMessage };
}
