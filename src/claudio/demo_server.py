
import http.server
import socketserver
import json
import threading
import time
import sys
import os

# Add the directory containing realtime_engine.py and quality_config.py to the Python path
from .realtime_engine import RealtimeEngine, AudioProfile

class ClaudioDemoServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/analysis_results':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Retrieve the latest analysis results from the RealtimeEngine
            if hasattr(self.server, 'realtime_engine') and self.server.realtime_engine:
                results = self.server.realtime_engine.get_analysis_results()
                self.wfile.write(json.dumps(results).encode('utf-8'))
            else:
                self.wfile.write(json.dumps({}).encode('utf-8'))
        else:
            # Serve static files from the current directory (claudio/src/claudio)
            super().do_GET()

def run_realtime_engine_thread(engine):
    # This function will run the RealtimeEngine in a separate thread
    # For demonstration, we'll just keep it running and generating (simulated) data
    # In a real scenario, this would involve starting audio streams, etc.
    print("RealtimeEngine thread started.")
    engine.start()
    try:
        while True:
            # Simulate processing audio and updating results
            # In a real setup, _audio_callback would be triggered by PyAudio
            # and update _analysis_results internally.
            # Here we just ensure the thread stays alive.
            time.sleep(1)
    except KeyboardInterrupt:
        print("RealtimeEngine thread stopped.")
    finally:
        engine.stop()

def start_demo_server(port=8000):
    Handler = ClaudioDemoServer
    # Ensure the handler serves files from the correct directory
    Handler.directory = 'claudio/src/claudio'

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving Claudio Real-Time Proof Dashboard at http://localhost:{port}/realtime_dashboard.html")

        # Initialize and start the RealtimeEngine in a separate thread
        # For a full benchmark, you would load reference audio here.
        # For now, we'll start with a basic configuration.
        realtime_engine = RealtimeEngine(profile=AudioProfile.BENCHMARK)
        httpd.realtime_engine = realtime_engine # Attach engine to server for access in handler

        engine_thread = threading.Thread(target=run_realtime_engine_thread, args=(realtime_engine,))
        engine_thread.daemon = True # Allow main program to exit even if thread is running
        engine_thread.start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
        finally:
            realtime_engine.stop()
            engine_thread.join() # Wait for engine thread to finish

if __name__ == '__main__':
    start_demo_server()
