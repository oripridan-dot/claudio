import pytest
from fastapi.testclient import TestClient

from claudio.server import auth
from claudio.server.claudio_server import app

client = TestClient(app)


def test_auth_token_generation():
    response = client.post("/api/auth/token", json={"username": "TestMusician"})
    assert response.status_code == 200
    data = response.json()
    assert "token" in data

    # Verify token payload
    payload = auth.verify_token(data["token"])
    assert payload is not None
    assert payload["sub"] == "TestMusician"


def test_unauthenticated_ws_rejection():
    # Attempt connecting without a token
    with pytest.raises(Exception) as excinfo, client.websocket_connect("/ws/collab/room123"):
        pass

    assert excinfo.value.code in (1008, 403)


def test_authenticated_ws_connection():
    # 1. Create a room first
    room_resp = client.post("/api/collab/create")
    room_id = room_resp.json()["room_id"]

    # 2. Get token
    resp = client.post("/api/auth/token", json={"username": "ValidUser"})
    token = resp.json()["token"]

    # 3. Connect with token to valid room
    with client.websocket_connect(f"/ws/collab/{room_id}?token={token}") as websocket:
        # Should receive the peer_joined packet first, then welcome
        data = websocket.receive_json()
        assert data["type"] == "peer_joined"

        data2 = websocket.receive_json()
        assert data2["type"] == "welcome"
        assert data2["room_id"] == room_id
        assert "peer_id" in data2
