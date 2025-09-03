"""Extended tests for utils modules to improve coverage."""

from lite_agent.utils.metrics import TimingMetrics


class TestMetrics:
    """Test metrics utility functions."""

    def test_calculate_latency_ms_basic(self):
        """Test basic calculate_latency_ms functionality."""
        from datetime import datetime, timezone

        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        first_output_time = datetime(2023, 1, 1, 12, 0, 0, 150000, tzinfo=timezone.utc)  # 150ms later

        result = TimingMetrics.calculate_latency_ms(start_time, first_output_time)

        # Should calculate correct latency
        assert result == 150

    def test_calculate_latency_ms_none_input(self):
        """Test calculate_latency_ms with None input."""
        from datetime import datetime, timezone

        result = TimingMetrics.calculate_latency_ms(None, None)
        assert result is None

        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = TimingMetrics.calculate_latency_ms(start_time, None)
        assert result is None

        first_output_time = datetime(2023, 1, 1, 12, 0, 0, 150000, tzinfo=timezone.utc)
        result = TimingMetrics.calculate_latency_ms(None, first_output_time)
        assert result is None

    def test_calculate_output_time_ms_basic(self):
        """Test basic calculate_output_time_ms functionality."""
        from datetime import datetime, timezone

        first_output_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        complete_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)  # 1 second later

        result = TimingMetrics.calculate_output_time_ms(first_output_time, complete_time)

        # Should calculate correct output time
        assert result == 1000

    def test_calculate_output_time_ms_none_input(self):
        """Test calculate_output_time_ms with None input."""
        result = TimingMetrics.calculate_output_time_ms(None, None)
        assert result is None

    def test_calculate_total_time_ms_basic(self):
        """Test basic calculate_total_time_ms functionality."""
        from datetime import datetime, timezone

        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        complete_time = datetime(2023, 1, 1, 12, 0, 2, tzinfo=timezone.utc)  # 2 seconds later

        result = TimingMetrics.calculate_total_time_ms(start_time, complete_time)

        # Should calculate correct total time
        assert result == 2000

    def test_calculate_total_time_ms_none_input(self):
        """Test calculate_total_time_ms with None input."""
        result = TimingMetrics.calculate_total_time_ms(None, None)
        assert result is None

    def test_timing_metrics_zero_duration(self):
        """Test timing metrics with zero duration."""
        from datetime import datetime, timezone

        same_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Same time for start and end should give 0
        result = TimingMetrics.calculate_total_time_ms(same_time, same_time)
        assert result == 0

        result = TimingMetrics.calculate_latency_ms(same_time, same_time)
        assert result == 0

        result = TimingMetrics.calculate_output_time_ms(same_time, same_time)
        assert result == 0

    def test_timing_metrics_with_microseconds(self):
        """Test timing metrics with microsecond precision."""
        from datetime import datetime, timezone

        start_time = datetime(2023, 1, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc)  # 500ms later

        result = TimingMetrics.calculate_latency_ms(start_time, end_time)

        # Should handle microsecond precision
        assert result == 500
