import json

import pytest

from metaflow.plugins.datatools.s3.s3 import S3Object, RangeInfo


class TestS3ObjectBasicProperties:
    """Test S3Object constructed without a local file path."""

    def test_properties_with_prefix_and_url(self):
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/subdir/file.txt",
            path=None,
            size=1024,
            content_type="text/plain",
            metadata={"metaflow-user-attributes": '{"author": "test"}'},
            range_info=RangeInfo(
                total_size=2048, request_offset=0, request_length=1024
            ),
            last_modified=1700000000,
            encryption="AES256",
        )
        assert obj.url == "s3://bucket/root/subdir/file.txt"
        assert obj.prefix == "s3://bucket/root"
        assert obj.key == "subdir/file.txt"
        assert obj.size == 1024
        assert obj.exists is True
        assert obj.downloaded is False
        assert obj.path is None
        assert obj.content_type == "text/plain"
        assert obj.metadata == {"author": "test"}
        assert obj.range_info == RangeInfo(2048, 0, 1024)
        assert obj.last_modified == 1700000000
        assert obj.encryption == "AES256"
        assert obj.has_info is True

    def test_properties_without_optional_fields(self):
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/key",
            path=None,
        )
        assert obj.url == "s3://bucket/root/key"
        assert obj.key == "key"
        assert obj.prefix == "s3://bucket/root"
        assert obj.size is None
        assert obj.exists is False
        assert obj.downloaded is False
        assert obj.content_type is None
        assert obj.metadata is None
        assert obj.range_info is None
        assert obj.last_modified is None
        assert obj.encryption is None
        assert obj.has_info is False

    def test_blob_and_text_none_without_path(self):
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None)
        assert obj.blob is None
        assert obj.text is None


class TestS3ObjectWithLocalFile:
    """Test S3Object when a local file path is provided."""

    def test_blob_returns_file_contents(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/data.bin",
            path=str(p),
        )
        assert obj.blob == b"hello world"
        assert obj.size == 11
        assert obj.exists is True
        assert obj.downloaded is True

    def test_text_returns_decoded_string(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_bytes("héllo wörld".encode("utf-8"))
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/data.txt",
            path=str(p),
        )
        assert obj.text == "héllo wörld"

    def test_size_from_file_overrides_constructor_size(self, tmp_path):
        p = tmp_path / "small.txt"
        p.write_bytes(b"abc")
        obj = S3Object(
            prefix=None,
            url="s3://bucket/small.txt",
            path=str(p),
            size=9999,
        )
        # size should come from os.stat, not the constructor arg
        assert obj.size == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        obj = S3Object(
            prefix=None,
            url="s3://bucket/empty.txt",
            path=str(p),
        )
        assert obj.size == 0
        assert obj.blob == b""
        assert obj.text == ""
        assert obj.exists is True
        assert obj.downloaded is True


class TestS3ObjectKeyPrefixLogic:
    """Test the key/prefix derivation logic in __init__."""

    def test_prefix_none_sets_key_to_url(self):
        obj = S3Object(prefix=None, url="s3://bucket/full/path", path=None)
        assert obj.key == "s3://bucket/full/path"
        assert obj.prefix is None

    def test_prefix_equals_url(self):
        obj = S3Object(
            prefix="s3://bucket/path",
            url="s3://bucket/path",
            path=None,
        )
        assert obj.key == "s3://bucket/path"
        assert obj.prefix is None

    def test_prefix_strips_trailing_slash(self):
        obj = S3Object(
            prefix="s3://bucket/root/",
            url="s3://bucket/root/subkey",
            path=None,
        )
        assert obj.key == "subkey"
        assert obj.prefix == "s3://bucket/root/"

    def test_nested_key_extraction(self):
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/a/b/c.txt",
            path=None,
        )
        assert obj.key == "a/b/c.txt"

    def test_key_trailing_slash_stripped(self):
        obj = S3Object(
            prefix="s3://bucket/root",
            url="s3://bucket/root/dir/",
            path=None,
        )
        assert obj.key == "dir"


class TestS3ObjectMetadata:
    """Test metadata parsing from the metaflow-user-attributes key."""

    def test_metadata_parsed_from_user_attributes(self):
        attrs = {"key1": "value1", "key2": 42}
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            metadata={"metaflow-user-attributes": json.dumps(attrs)},
        )
        assert obj.metadata == attrs

    def test_metadata_none_when_no_user_attributes_key(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            metadata={"other-key": "value"},
        )
        assert obj.metadata is None

    def test_metadata_none_when_metadata_is_none(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            metadata=None,
        )
        assert obj.metadata is None

    def test_empty_user_attributes(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            metadata={"metaflow-user-attributes": "{}"},
        )
        assert obj.metadata == {}


class TestS3ObjectRangeInfo:
    """Test range_info normalization in __init__."""

    def test_range_info_passthrough(self):
        ri = RangeInfo(total_size=1000, request_offset=100, request_length=200)
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, range_info=ri)
        assert obj.range_info == ri

    def test_range_info_none(self):
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, range_info=None)
        assert obj.range_info is None

    def test_negative_request_length_normalized_to_total_size(self):
        ri = RangeInfo(total_size=500, request_offset=10, request_length=-1)
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, range_info=ri)
        assert obj.range_info.request_length == 500
        assert obj.range_info.request_offset == 10
        assert obj.range_info.total_size == 500

    def test_none_request_length_normalized_to_total_size(self):
        ri = RangeInfo(total_size=800, request_offset=0, request_length=None)
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, range_info=ri)
        assert obj.range_info.request_length == 800
        assert obj.range_info.total_size == 800

    def test_zero_request_length_kept(self):
        ri = RangeInfo(total_size=100, request_offset=0, request_length=0)
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, range_info=ri)
        assert obj.range_info.request_length == 0


class TestS3ObjectHasInfo:
    """Test the has_info property for various combinations."""

    def test_has_info_with_content_type(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            content_type="application/json",
        )
        assert obj.has_info is True

    def test_has_info_with_metadata(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            metadata={"metaflow-user-attributes": '{"a": 1}'},
        )
        assert obj.has_info is True

    def test_has_info_with_range_info(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            range_info=RangeInfo(100, 0, 100),
        )
        assert obj.has_info is True

    def test_has_info_with_encryption(self):
        obj = S3Object(
            prefix=None,
            url="s3://bucket/key",
            path=None,
            encryption="aws:kms",
        )
        assert obj.has_info is True

    def test_has_info_false_with_nothing(self):
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None)
        assert obj.has_info is False


class TestS3ObjectStrRepr:
    """Test __str__ and __repr__ output."""

    def test_str_with_local_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"data")
        obj = S3Object(prefix=None, url="s3://bucket/f.txt", path=str(p))
        s = str(obj)
        assert "s3://bucket/f.txt" in s
        assert "4 bytes" in s
        assert "local" in s

    def test_str_with_size_no_path(self):
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, size=256)
        s = str(obj)
        assert "s3://bucket/key" in s
        assert "256 bytes" in s
        assert "in S3" in s

    def test_str_nonexistent(self):
        obj = S3Object(prefix=None, url="s3://bucket/missing", path=None)
        s = str(obj)
        assert "s3://bucket/missing" in s
        assert "does not exist" in s

    def test_repr_equals_str(self):
        obj = S3Object(prefix=None, url="s3://bucket/key", path=None, size=10)
        assert repr(obj) == str(obj)
