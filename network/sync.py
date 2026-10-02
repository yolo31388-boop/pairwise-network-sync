"""网络状态同步 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class PlayerState:
    pid: str
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

class NetworkSync:
    def __init__(self):
        self.states: dict[str, PlayerState] = {}
        self.prediction_log: list = []  # bug2: 不回滚

    def send_position(self, pid: str, x: float, y: float) -> dict:
        # bug1: 不插值，直接瞬移
        if pid not in self.states:
            self.states[pid] = PlayerState(pid)
        self.states[pid].x = x
        self.states[pid].y = y
        return {"x": x, "y": y}

    def client_prediction(self, pid: str, action: str) -> bool:
        # bug2: 服务器纠正后不回滚
        self.prediction_log.append({"pid": pid, "action": action})
        return True

    def compress_state(self, state: PlayerState) -> dict:
        # bug3: 浮点数压成整数，精度不够
        return {"x": int(state.x), "y": int(state.y)}

    def lag_compensation(self, attacker_latency: int, target_latency: int) -> float:
        # bug4: 只补偿攻击者
        return attacker_latency / 1000.0

    def reconnect_sync(self, pid: str) -> dict:
        # bug5: 全量同步不增量
        return {pid: self.states.get(pid, PlayerState(pid))}
