import re

with open("../claudio/frontend/src/engine/IntentEngine.ts", "r") as f:
    code = f.read()

# Make all WebRTC/Network properties public
code = code.replace("private pc:", "pc:")
code = code.replace("private dataChannel:", "dataChannel:")
code = code.replace("private remoteStream:", "remoteStream:")
code = code.replace("private remoteStreamSource:", "remoteStreamSource:")
code = code.replace("private reconnectAttempts", "reconnectAttempts")
code = code.replace("private maxReconnectAttempts", "maxReconnectAttempts")
code = code.replace("private reconnectTimer:", "reconnectTimer:")
code = code.replace("private serverUrl", "serverUrl")
code = code.replace("private displayName", "displayName")
code = code.replace("private jwtToken:", "jwtToken:")
code = code.replace("private lastPingTs", "lastPingTs")
code = code.replace("private remoteSeq:", "remoteSeq:")
code = code.replace("private packetsReceived", "packetsReceived")
code = code.replace("private packetsLost", "packetsLost")
code = code.replace("private packetsWindowReceived", "packetsWindowReceived")
code = code.replace("private packetsWindowLost", "packetsWindowLost")
code = code.replace("private lastNetworkUpdateTs", "lastNetworkUpdateTs")
code = code.replace("private latestRemoteMelBands", "latestRemoteMelBands")
code = code.replace("private ws:", "ws:")
code = code.replace("private nextAudioTime", "nextAudioTime")

# We will extract _connectWebSocket
block_start = "  _connectWebSocket(): void {"
block_end = "  private _startPing(): void {"
code = code.replace("private _connectWebSocket", "_connectWebSocket")

pattern = r"(  _connectWebSocket\(\): void \{.*?\n  \})\n\n(  private _startPing)"
match = re.search(pattern, code, re.DOTALL)

if match:
    ws_func = match.group(1)
    
    # We create IntentWebRTC.ts bridging this function
    webrtc_ts = """import type { IntentEngine } from './IntentEngine';
import type { IntentFrame, PivotFrame } from './types';
import { decodePacket } from './protocol';

export function initIntentWebSocket(engine: any) {
""" + ws_func.replace("this.", "engine.").replace("  _connectWebSocket(): void {", "")
    
    with open("../claudio/frontend/src/engine/IntentWebRTC.ts", "w") as f:
         f.write(webrtc_ts)

    # Now replace the block in IntentEngine
    new_connect = """  _connectWebSocket(): void {
    initIntentWebSocket(this);
  }"""
    code = code[:match.start()] + new_connect + "\n\n" + match.group(2) + code[match.end():]

# Also extract _initWebRTC
pattern_rt = r"(  private async _initWebRTC\(\): Promise<void> \{.*?\n  \})\n\n(  private _scheduleAudioBlock)"
match_rt = re.search(pattern_rt, code, re.DOTALL)

if match_rt:
    rt_func = match_rt.group(1)
    rt_ts = """
export async function initIntentWebRTC(engine: any) {
""" + rt_func.replace("this.", "engine.").replace("  private async _initWebRTC(): Promise<void> {", "")
    with open("../claudio/frontend/src/engine/IntentWebRTC.ts", "a") as f:
         f.write(rt_ts)

    new_rt = """  async _initWebRTC(): Promise<void> {
    await initIntentWebRTC(this);
  }"""
    code = code[:match_rt.start()] + new_rt + "\n\n" + match_rt.group(2) + code[match_rt.end():]

# Add imports for these externalized functions
import_stmt = "import { initIntentWebSocket, initIntentWebRTC } from './IntentWebRTC';\n"
code = code.replace("import { encodePacket, decodePacket }", import_stmt + "import { encodePacket, decodePacket }")

with open("../claudio/frontend/src/engine/IntentEngine.ts", "w") as f:
    f.write(code)

