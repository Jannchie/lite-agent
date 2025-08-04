"""
简化的 stream handlers 测试，重点测试代码覆盖率
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from lite_agent.stream_handlers.litellm import ensure_record_file


class TestEnsureRecordFile:
    """测试ensure_record_file函数"""

    def test_ensure_record_file_none(self):
        """测试传入None的情况"""
        result = ensure_record_file(None)
        assert result is None

    def test_ensure_record_file_existing_directory(self):
        """测试目录已存在的情况"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            record_file = temp_path / "test.jsonl"
            result = ensure_record_file(record_file)
            assert result == record_file

    def test_ensure_record_file_create_directory(self):
        """测试需要创建目录的情况"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_subdir"
            record_file = new_dir / "test.jsonl"

            # 目录不存在
            assert not new_dir.exists()

            result = ensure_record_file(record_file)

            # 目录应该被创建
            assert new_dir.exists()
            assert result == record_file

    @patch("lite_agent.stream_handlers.litellm.logger")
    def test_ensure_record_file_logs_warning(self, mock_logger):
        """测试创建目录时记录警告日志"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_subdir"
            record_file = new_dir / "test.jsonl"

            ensure_record_file(record_file)

            # 应该记录警告日志
            mock_logger.warning.assert_called_once()
            assert "does not exist, creating it" in str(mock_logger.warning.call_args)
