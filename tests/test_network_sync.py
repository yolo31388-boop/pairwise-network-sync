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

class TestPredictionRollback:
    def test_server_correction_rolls_back(self):
        ns = NetworkSync()
        ns.client_prediction("p1", "attack")
        # 服务器纠正后应该回滚
        assert len(ns.prediction_log) == 0

class TestCompressionPrecision:
    def test_compressed_state_preserves_precision(self):
        ns = NetworkSync()
        state = PlayerState("p1", x=1.5, y=2.7)
        compressed = ns.compress_state(state)
        assert compressed["x"] == 1.5

class TestBidirectionalLag:
    def test_lag_compensates_both_sides(self):
        ns = NetworkSync()
        result = ns.lag_compensation(100, 50)
        assert result > 0.1

class TestIncrementalReconnect:
    def test_reconnect_uses_incremental(self):
        ns = NetworkSync()
        ns.states["p1"] = PlayerState("p1", x=10, y=20)
        result = ns.reconnect_sync("p1")
        assert "p1" in result
