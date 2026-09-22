import base64
import json
import re
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mod_explore.map_export import (
    generate_editor, pointcloud_xyzi, write_pcd, write_pcd_from_stream,
)


def field(name, offset, datatype=7):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


@pytest.mark.parametrize('endian', ['<', '>'])
def test_organized_cloud_with_padding_and_reordered_fields(endian):
    # Rows and points have padding; field declaration order differs from byte order.
    records = [(3., 1., 2., 42), (6., 4., 5., 255)]
    data = b''.join(struct.pack(endian + 'fffH6x', *p) + b'padding!' for p in records)
    msg = SimpleNamespace(width=1, height=2, point_step=20, row_step=28,
                          fields=[field('intensity', 12, 4), field('x', 4),
                                  field('z', 0), field('y', 8)],
                          is_bigendian=endian == '>', data=data)
    np.testing.assert_array_equal(pointcloud_xyzi(msg), [[1, 2, 3, 42], [4, 5, 6, 255]])


def test_invalid_xyz_removed_and_missing_intensity_defaults_to_zero():
    msg = SimpleNamespace(width=3, height=1, point_step=12, row_step=36,
                          fields=[field('x', 0), field('y', 4), field('z', 8)],
                          is_bigendian=False,
                          data=struct.pack('<9f', 1, 2, 3, float('nan'), 0, 0, 0, float('inf'), 0))
    np.testing.assert_array_equal(pointcloud_xyzi(msg), [[1, 2, 3, 0]])
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError, match='Truncated'):
        pointcloud_xyzi(msg)


def test_malformed_field_rejected():
    msg = SimpleNamespace(width=1, height=1, point_step=12, row_step=12,
                          fields=[field('x', 0), field('y', 4)],
                          is_bigendian=False, data=bytes(12))
    with pytest.raises(ValueError, match='x, y and z'):
        pointcloud_xyzi(msg)
    msg.fields.append(field('z', 12))
    with pytest.raises(ValueError, match='offset'):
        pointcloud_xyzi(msg)


def test_binary_pcd_and_embedded_html_preserve_full_payload(tmp_path):
    points = np.array([[1.25, -2.5, 3.75, 42], [0, 10, -5, 255]], dtype='<f4')
    pcd = tmp_path / 'map.pcd'
    html = tmp_path / 'map.html'
    write_pcd(pcd, points)
    header, payload = pcd.read_bytes().split(b'DATA binary\n', 1)
    assert b'FIELDS x y z intensity\n' in header
    assert b'WIDTH 2\nHEIGHT 1\n' in header
    assert b'POINTS 2\n' in header
    assert struct.unpack('<8f', payload) == (1.25, -2.5, 3.75, 42, 0, 10, -5, 255)
    frame = '</script><script>bad()</script>'
    generate_editor(pcd, html, frame)
    content = html.read_text()
    data = re.search(r'<script[^>]*id="map-points"[^>]*>(.*?)</script>', content, re.S).group(1)
    meta = re.search(r'<script[^>]*id="map-meta"[^>]*>(.*?)</script>', content, re.S).group(1)
    assert base64.b64decode(data) == payload
    assert json.loads(meta)['frame_id'] == frame
    assert frame not in content
    assert '__POINT_DATA_BASE64__' not in content


def test_stream_pcd_and_empty_or_truncated_exports(tmp_path):
    raw = tmp_path / 'raw'
    raw.write_bytes(struct.pack('<4f', 1, 2, 3, 4))
    target = tmp_path / 'map.pcd'
    write_pcd_from_stream(target, raw, 1)
    original = target.read_bytes()
    with pytest.raises(ValueError, match='Point count'):
        write_pcd_from_stream(target, raw, 2)
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match='empty'):
        write_pcd(target, np.empty((0, 4)))
    target.write_bytes(original[:-1])
    with pytest.raises(ValueError, match='point count'):
        generate_editor(target, tmp_path / 'bad.html')
