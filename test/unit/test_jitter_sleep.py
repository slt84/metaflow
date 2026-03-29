"""Unit tests for S3._jitter_sleep: exponential backoff with jitter."""

from unittest.mock import MagicMock, patch

import pytest

from metaflow.plugins.datatools.s3.s3 import S3


def _make_s3():
    """Create an S3 instance without triggering __init__ (which needs boto3/tmpdir)."""
    obj = object.__new__(S3)
    return obj


class TestJitterSleepExponentialScaling:
    """Verify the backoff interval scales as base**trynum."""

    @pytest.mark.parametrize(
        "trynum, base, expected_base_interval",
        [
            (0, 2, 1),  # 2**0 = 1
            (1, 2, 2),  # 2**1 = 2
            (2, 2, 4),  # 2**2 = 4
            (3, 2, 8),  # 2**3 = 8
            (5, 2, 32),  # 2**5 = 32
            (8, 2, 256),  # 2**8 = 256
            (0, 3, 1),  # 3**0 = 1
            (4, 3, 81),  # 3**4 = 81
        ],
    )
    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_exponential_formula_with_zero_jitter(
        self, mock_uniform, mock_sleep, trynum, base, expected_base_interval
    ):
        """With jitter contribution = 0, sleep value should equal base**trynum exactly."""
        s3 = _make_s3()
        s3._jitter_sleep(trynum, base=base, cap=360)

        mock_sleep.assert_called_once_with(expected_base_interval)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_intervals_increase_monotonically(self, mock_uniform, mock_sleep):
        """Successive trynum values should yield strictly increasing sleep times
        (until the cap is hit)."""
        s3 = _make_s3()
        intervals = []
        for trynum in range(9):  # 2**8 = 256, still below cap of 360
            mock_sleep.reset_mock()
            s3._jitter_sleep(trynum, base=2, cap=360)
            intervals.append(mock_sleep.call_args[0][0])

        for i in range(1, len(intervals)):
            assert (
                intervals[i] > intervals[i - 1]
            ), "interval[%d]=%s should be > interval[%d]=%s" % (
                i,
                intervals[i],
                i - 1,
                intervals[i - 1],
            )

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=1.0)
    def test_positive_jitter_adds_to_interval(self, mock_uniform, mock_sleep):
        """When jitter is maximally positive, the sleep should be interval * (1 + jitter)."""
        s3 = _make_s3()
        # trynum=3, base=2 => interval=8, jitter=0.1 => 8 + 8*0.1*1.0 = 8.8
        s3._jitter_sleep(3, base=2, cap=360, jitter=0.1)

        mock_sleep.assert_called_once_with(8.8)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_negative_jitter_subtracts_from_interval(self, mock_uniform, mock_sleep):
        """When jitter is maximally negative, the sleep should be interval * (1 - jitter)."""
        s3 = _make_s3()
        # trynum=3, base=2 => interval=8, jitter=0.1 => 8 + 8*0.1*(-1.0) = 7.2
        s3._jitter_sleep(3, base=2, cap=360, jitter=0.1)

        mock_sleep.assert_called_once_with(7.2)


class TestJitterSleepCap:
    """Verify the 360-second cap is strictly respected."""

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_cap_applied_at_boundary(self, mock_uniform, mock_sleep):
        """When base**trynum exceeds cap, interval should be capped."""
        s3 = _make_s3()
        # 2**9 = 512, which exceeds default cap of 360
        s3._jitter_sleep(9, base=2, cap=360)

        mock_sleep.assert_called_once_with(360)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_cap_applied_at_very_large_trynum(self, mock_uniform, mock_sleep):
        """Even at very large retry numbers, cap is respected."""
        s3 = _make_s3()
        s3._jitter_sleep(100, base=2, cap=360)

        mock_sleep.assert_called_once_with(360)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=1.0)
    def test_cap_with_max_positive_jitter(self, mock_uniform, mock_sleep):
        """Cap + maximum positive jitter: sleep = cap * (1 + jitter)."""
        s3 = _make_s3()
        # 2**9 = 512 => capped to 360, then 360 + 360*0.1*1.0 = 396.0
        s3._jitter_sleep(9, base=2, cap=360, jitter=0.1)

        mock_sleep.assert_called_once_with(396.0)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_cap_with_max_negative_jitter(self, mock_uniform, mock_sleep):
        """Cap + maximum negative jitter: sleep = cap * (1 - jitter)."""
        s3 = _make_s3()
        # capped to 360, then 360 + 360*0.1*(-1.0) = 324.0
        s3._jitter_sleep(9, base=2, cap=360, jitter=0.1)

        mock_sleep.assert_called_once_with(324.0)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_custom_cap_is_respected(self, mock_uniform, mock_sleep):
        """A non-default cap value should still be respected."""
        s3 = _make_s3()
        # 2**10 = 1024 > cap=100
        s3._jitter_sleep(10, base=2, cap=100)

        mock_sleep.assert_called_once_with(100)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=0.0)
    def test_exactly_at_cap_boundary(self, mock_uniform, mock_sleep):
        """When base**trynum equals cap exactly, interval should be cap."""
        s3 = _make_s3()
        # cap=256, 2**8 = 256 => exactly at cap
        s3._jitter_sleep(8, base=2, cap=256)

        mock_sleep.assert_called_once_with(256)


class TestJitterSleepNonNegative:
    """Verify sleep time is never negative."""

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_trynum_zero_max_negative_jitter(self, mock_uniform, mock_sleep):
        """trynum=0 gives interval=1; with max negative jitter (0.1): 1 - 0.1 = 0.9 >= 0."""
        s3 = _make_s3()
        s3._jitter_sleep(0, base=2, cap=360, jitter=0.1)

        sleep_val = mock_sleep.call_args[0][0]
        assert sleep_val >= 0, "Sleep value %s is negative" % sleep_val

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_large_jitter_clamped_to_zero(self, mock_uniform, mock_sleep):
        """With jitter > 1.0 and max negative uniform, result should be clamped to 0."""
        s3 = _make_s3()
        # interval=1, jitter=2.0, uniform=-1 => 1 + 1*2.0*(-1) = -1.0 => clamped to 0
        s3._jitter_sleep(0, base=2, cap=360, jitter=2.0)

        mock_sleep.assert_called_once_with(0)

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_jitter_exactly_one_clamped_to_zero(self, mock_uniform, mock_sleep):
        """With jitter=1.0 and uniform=-1: interval + interval*1.0*(-1) = 0."""
        s3 = _make_s3()
        # interval=1 (trynum=0), jitter=1.0 => 1 + 1*1.0*(-1) = 0
        s3._jitter_sleep(0, base=2, cap=360, jitter=1.0)

        mock_sleep.assert_called_once_with(0)

    @pytest.mark.parametrize("jitter_val", [-1.0, -0.5, 0.0, 0.5, 1.0])
    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    def test_non_negative_across_jitter_range(self, mock_sleep, jitter_val):
        """Across a range of uniform outputs, sleep should never go negative."""
        s3 = _make_s3()
        with patch(
            "metaflow.plugins.datatools.s3.s3.random.uniform", return_value=jitter_val
        ):
            for trynum in range(20):
                mock_sleep.reset_mock()
                s3._jitter_sleep(trynum, base=2, cap=360, jitter=0.1)

                sleep_val = mock_sleep.call_args[0][0]
                assert (
                    sleep_val >= 0
                ), "Negative sleep %s at trynum=%d, jitter_val=%s" % (
                    sleep_val,
                    trynum,
                    jitter_val,
                )

    @patch("metaflow.plugins.datatools.s3.s3.time.sleep")
    @patch("metaflow.plugins.datatools.s3.s3.random.uniform", return_value=-1.0)
    def test_zero_cap_with_negative_jitter(self, mock_uniform, mock_sleep):
        """cap=0 means interval=0; with any jitter, should clamp to 0."""
        s3 = _make_s3()
        s3._jitter_sleep(5, base=2, cap=0, jitter=0.5)

        mock_sleep.assert_called_once_with(0)
