import socket
import ipaddress
import pytest

@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    original=socket.socket.connect
    def deny(sock, address):
        # Windows asyncio creates its self-pipe via a loopback socket pair.
        if isinstance(address,tuple):
            try:
                if ipaddress.ip_address(address[0]).is_loopback:
                    return original(sock,address)
            except ValueError:
                pass
        raise AssertionError("Offline tests must not access external networks")
    monkeypatch.setattr(socket.socket, "connect", deny)
