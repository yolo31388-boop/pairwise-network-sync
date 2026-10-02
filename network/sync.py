"""网络状态同步"""
import struct
import zlib
from dataclasses import dataclass


@dataclass
class PlayerState:
    pid: str
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


def _half(value: float) -> float:
    """Round-trip a float through IEEE 754 half precision (binary16)."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


class NetworkSync:
    def __init__(self, sync_rate_hz: int = 60, interp_alpha: float = 0.5):
        self.states: dict[str, PlayerState] = {}
        self.prediction_log: list = []
        self.sync_rate_hz = sync_rate_hz    # 同步频率可配置，不再写死 10Hz
        self.interp_alpha = interp_alpha    # 位置插值平滑系数
        self.state_version: dict[str, int] = {}
        self.synced_version: dict[str, dict[str, int]] = {}
        self._seq = 0

    def _touch(self, pid: str) -> None:
        self.state_version[pid] = self.state_version.get(pid, 0) + 1

    def send_position(self, pid: str, x: float, y: float) -> dict:
        # 插值平滑移动：向服务器位置逼近，而不是直接瞬移
        if pid not in self.states:
            self.states[pid] = PlayerState(pid)
        state = self.states[pid]
        state.x += (x - state.x) * self.interp_alpha
        state.y += (y - state.y) * self.interp_alpha
        self._touch(pid)
        return {"x": state.x, "y": state.y}

    def client_prediction(self, pid: str, action: str) -> bool:
        # 客户端预测后与服务器对账，被纠正的预测回滚出日志
        self._seq += 1
        self.prediction_log.append({"pid": pid, "action": action, "seq": self._seq})
        self.server_correction(pid, self._seq)
        return True

    def server_correction(self, pid: str, last_seq: int) -> None:
        # 服务器纠正：回滚（移除）该玩家已结算的预测动作
        self.prediction_log = [
            entry for entry in self.prediction_log
            if not (entry["pid"] == pid and entry["seq"] <= last_seq)
        ]

    def compress_state(self, state: PlayerState) -> dict:
        # 半精度浮点压缩 + CRC32 校验和
        x = _half(state.x)
        y = _half(state.y)
        payload = struct.pack("<ee", x, y)
        return {"x": x, "y": y, "checksum": zlib.crc32(payload) & 0xFFFFFFFF}

    def verify_state(self, packet: dict) -> bool:
        # 校验数据包完整性，损坏的包拒绝接受
        try:
            payload = struct.pack("<ee", _half(packet["x"]), _half(packet["y"]))
        except (KeyError, TypeError, struct.error):
            return False
        return (zlib.crc32(payload) & 0xFFFFFFFF) == packet.get("checksum")

    def lag_compensation(self, attacker_latency: int, target_latency: int) -> float:
        # 双向延迟补偿：攻击者和被攻击者的延迟都计入
        return (attacker_latency + target_latency) / 1000.0

    def reconnect_sync(self, pid: str) -> dict:
        # 增量同步：只传该玩家上次同步之后有变化的状态
        last_seen = self.synced_version.get(pid, {})
        delta = {}
        for other, state in self.states.items():
            version = self.state_version.get(other, 0)
            if last_seen.get(other, -1) < version:
                delta[other] = state
        self.synced_version[pid] = {
            other: self.state_version.get(other, 0) for other in self.states
        }
        return delta
