from datetime import datetime


class TimingMetrics:
    """Utility class for calculating timing metrics in LLM responses."""

    @staticmethod
    def calculate_latency_ms(start_time: datetime | None, first_output_time: datetime | None) -> int | None:
        """Calculate latency from start to first output.
        
        Args:
            start_time: When the request started
            first_output_time: When the first output was received
            
        Returns:
            Latency in milliseconds, or None if either time is missing
        """
        if start_time is None or first_output_time is None:
            return None
        return int((first_output_time - start_time).total_seconds() * 1000)

    @staticmethod
    def calculate_output_time_ms(first_output_time: datetime | None, output_complete_time: datetime | None) -> int | None:
        """Calculate time from first output to completion.
        
        Args:
            first_output_time: When the first output was received
            output_complete_time: When output was completed
            
        Returns:
            Output time in milliseconds, or None if either time is missing
        """
        if first_output_time is None or output_complete_time is None:
            return None
        return int((output_complete_time - first_output_time).total_seconds() * 1000)

    @staticmethod
    def calculate_total_time_ms(start_time: datetime | None, output_complete_time: datetime | None) -> int | None:
        """Calculate total time from start to completion.
        
        Args:
            start_time: When the request started
            output_complete_time: When output was completed
            
        Returns:
            Total time in milliseconds, or None if either time is missing
        """
        if start_time is None or output_complete_time is None:
            return None
        return int((output_complete_time - start_time).total_seconds() * 1000)
