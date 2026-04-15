"""
test_webrtc_integration.py — Bidirectional WebRTC/WS Integration Tests

Verifies that intents received via WebRTC are broadcasted to both WS and RTC peers.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from claudio.collab.webrtc_manager import WebRTCManager
from claudio.collab.session_manager import SessionManager

@pytest.mark.asyncio
async def test_webrtc_bidirectional_broadcast():
    # Setup
    session_manager = SessionManager()
    webrtc_manager = WebRTCManager(session_manager)
    
    room_id = await session_manager.create_room()
    
    # Mock WS peer
    ws_mock = MagicMock()
    ws_mock.send_bytes = AsyncMock()
    ws_mock.query_params = {}
    ws_peer = await session_manager.join_room(room_id, ws_mock, "WS_Peer")
    
    # Mock WebRTC peer (in WebRTCManager)
    rtc_peer_id = "rtc_123"
    rtc_channel_mock = MagicMock()
    rtc_channel_mock.readyState = "open"
    rtc_channel_mock.send = MagicMock()
    webrtc_manager.data_channels[rtc_peer_id] = rtc_channel_mock
    
    # We also need to add the rtc_peer to the session_manager room 
    # so that the broadcasters see it.
    rtc_ws_mock = MagicMock() # WebRTC peers still have a signaling WS
    rtc_ws_mock.send_bytes = AsyncMock()
    await session_manager.join_room(room_id, rtc_ws_mock, "RTC_Peer")
    # Join room returns a random id, let's just make sure we use the one it returns
    rtc_peer_info = list(session_manager.get_room(room_id).peers.values())[-1]
    rtc_peer_actual_id = rtc_peer_info.peer_id
    webrtc_manager.data_channels[rtc_peer_actual_id] = rtc_channel_mock

    # Test Action: Receive intent via WebRTC from rtc_peer
    test_data = b"test_intent_packet"
    await webrtc_manager.broadcast_intent_p2p(room_id, rtc_peer_actual_id, test_data)
    
    # Verify: The intent should NOT be sent back to the sender
    rtc_channel_mock.send.assert_not_called()
    
    # Verify: The intent should be sent to the WS peer
    # (In a real scenario, we'd call _broadcast_intent which orchestrates both)
    await webrtc_manager._broadcast_intent(rtc_peer_actual_id, room_id, test_data, None)
    
    # 1. Verification of WebSocket broadcast
    ws_mock.send_bytes.assert_called_with(test_data)
    
    # 2. Verification of WebRTC p2p broadcast (to other RTC peers if they existed)
    # Let's add another RTC peer to be sure
    rtc_peer_2_id = "rtc_456"
    rtc_channel_2_mock = MagicMock()
    rtc_channel_2_mock.readyState = "open"
    rtc_channel_2_mock.send = MagicMock()
    
    rtc_ws_2_mock = MagicMock()
    peer2 = await session_manager.join_room(room_id, rtc_ws_2_mock, "RTC_Peer_2")
    webrtc_manager.data_channels[peer2.peer_id] = rtc_channel_2_mock
    
    await webrtc_manager._broadcast_intent(rtc_peer_actual_id, room_id, test_data, None)
    rtc_channel_2_mock.send.assert_called_with(test_data)

    print("\n[OK] Bidirectional WebRTC/WS broadcast verified.")
