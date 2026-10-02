"""网络状态同步 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.sync import NetworkSync, PlayerState

class TestInterpolation:
    def test_position_smoothly_interpolates(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=0, y=0)
        ns.send_position("p1", 100, 100)
        # 应该有插值，不是直接到100
        assert ns.states["p1"].x <= 100

    def test_position_does_not_snap(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=0, y=0)
        ns.send_position("p1", 100, 100)
        # 一帧内只走一部分，不能瞬移到目标点
        assert 0 < ns.states["p1"].x < 100
        assert 0 < ns.states["p1"].y < 100

    def test_position_converges_to_target(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=0, y=0)
        for _ in range(20):
            ns.send_position("p1", 100, 100)
        # 多帧插值后应收敛到服务器位置
        assert abs(ns.states["p1"].x - 100) < 0.01
        assert abs(ns.states["p1"].y - 100) < 0.01

class TestPredictionRollback:
    def test_server_correction_rolls_back(self):
        ns = NetworkSync()
        ns.client_prediction("p1", "attack")
        # 服务器纠正后应该回滚
        assert len(ns.prediction_log) == 0

    def test_correction_only_rolls_back_target_player(self):
        ns = NetworkSync()
        ns.prediction_log.append({"pid": "p1", "action": "attack", "seq": 1})
        ns.prediction_log.append({"pid": "p2", "action": "jump", "seq": 2})
        ns.server_correction("p1", 1)
        # 只回滚被纠正玩家的预测，不波及其他玩家
        assert [entry["pid"] for entry in ns.prediction_log] == ["p2"]

    def test_correction_keeps_unconfirmed_predictions(self):
        ns = NetworkSync()
        ns.prediction_log.append({"pid": "p1", "action": "attack", "seq": 1})
        ns.prediction_log.append({"pid": "p1", "action": "jump", "seq": 2})
        ns.server_correction("p1", 1)
        # 服务器只确认到 seq=1，seq=2 必须保留待下次对账
        assert [entry["action"] for entry in ns.prediction_log] == ["jump"]

class TestCompressionPrecision:
    def test_compressed_state_preserves_precision(self):
        ns = NetworkSync()
        state = PlayerState("p1", x=1.5, y=2.7)
        compressed = ns.compress_state(state)
        assert compressed["x"] == 1.5

    def test_half_precision_error_is_bounded(self):
        ns = NetworkSync()
        compressed = ns.compress_state(PlayerState("p1", x=2.7, y=123.456))
        # 半精度误差应远小于整数截断的误差
        assert abs(compressed["x"] - 2.7) < 0.001
        assert abs(compressed["y"] - 123.456) < 0.1

    def test_checksum_detects_corruption(self):
        ns = NetworkSync()
        packet = ns.compress_state(PlayerState("p1", x=1.5, y=2.5))
        # 完好的包通过校验
        assert ns.verify_state(packet)
        # 数据被篡改后校验失败，拒绝接受
        packet["x"] = 99.0
        assert not ns.verify_state(packet)

class TestBidirectionalLag:
    def test_lag_compensates_both_sides(self):
        ns = NetworkSync()
        result = ns.lag_compensation(100, 50)
        assert result > 0.1

    def test_lag_compensation_is_symmetric(self):
        ns = NetworkSync()
        # 交换攻击者/被攻击者补偿值不变，且等于双方延迟之和
        assert ns.lag_compensation(100, 50) == ns.lag_compensation(50, 100)
        assert abs(ns.lag_compensation(100, 50) - 0.15) < 1e-9

class TestIncrementalReconnect:
    def test_reconnect_uses_incremental(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=10, y=20)
        result = ns.reconnect_sync("p1")
        assert "p1" in result

    def test_second_reconnect_sends_nothing(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=10, y=20)
        ns.reconnect_sync("p2")
        # 状态没有变化时，重连增量为空，不再全量下发
        assert ns.reconnect_sync("p2") == {}

    def test_reconnect_sends_only_changed_states(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=10, y=20)
        ns.states["p2"] = PlayerState("p2", x=1, y=2)
        ns.reconnect_sync("p3")
        ns.send_position("p1", 50, 50)
        delta = ns.reconnect_sync("p3")
        # 只有变化过的 p1 需要下发
        assert set(delta.keys()) == {"p1"}
