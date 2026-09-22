"""PCD and standalone editor export, independent of ROS runtime."""

import base64
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def pointcloud_xyzi(msg):
    return _pointcloud_xyz(msg, 'intensity')


def pointcloud_xyzrgb(msg):
    """Keep the published RGB bit pattern rather than converting it to intensity."""
    return _pointcloud_xyz(msg, 'rgb')


def _pointcloud_xyz(msg, color_field):
    """Read PointCloud2 fields, including organized rows and either byte order."""
    width, height = int(msg.width), int(msg.height)
    step, row_step = int(msg.point_step), int(msg.row_step)
    if width < 0 or height < 0 or step < 0 or row_step < width * step:
        raise ValueError('Invalid PointCloud2 dimensions or row stride')
    if width == 0 or height == 0:
        return np.empty((0, 4), dtype='<f4')
    if step == 0 or len(msg.data) < (height - 1) * row_step + width * step:
        raise ValueError('Truncated PointCloud2 data')
    fields = {field.name: field for field in msg.fields}
    if not all(name in fields for name in ('x', 'y', 'z')):
        raise ValueError('PointCloud2 must have x, y and z fields')
    if color_field == 'rgb' and 'rgb' not in fields:
        raise ValueError('The 2.5D map must contain its published rgb field')
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    endian = '>' if msg.is_bigendian else '<'
    names, formats, offsets = [], [], []
    for name in ('x', 'y', 'z', color_field):
        if name not in fields:
            continue
        field = fields[name]
        if field.datatype not in types or field.count != 1:
            raise ValueError(f'Unsupported PointCloud2 field {name}')
        dtype = np.dtype(endian + types[field.datatype])
        if name == 'rgb':
            if field.datatype not in (6, 7):
                raise ValueError('RGB must be packed FLOAT32 or UINT32')
            # Read bytes as an integer even when the field declares FLOAT32.
            dtype = np.dtype(endian + 'u4')
        if field.offset < 0 or field.offset + dtype.itemsize > step:
            raise ValueError(f'Invalid PointCloud2 offset for {name}')
        names.append(name)
        formats.append(dtype)
        offsets.append(field.offset)
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=step))
    cloud = np.ndarray((height, width), dtype=dtype, buffer=msg.data,
                       strides=(row_step, step))
    points = np.zeros((height * width, 4), dtype='<f4')
    with np.errstate(over='ignore', invalid='ignore'):
        for i, name in enumerate(('x', 'y', 'z')):
            if name in names:
                points[:, i] = cloud[name].reshape(-1)
        if color_field == 'rgb':
            points.view('<u4')[:, 3] = cloud['rgb'].reshape(-1)
        elif 'intensity' in names:
            points[:, 3] = cloud['intensity'].reshape(-1)
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    if color_field == 'intensity':
        points[~np.isfinite(points[:, 3]), 3] = 0
    return points


def pcd_header(point_count, frame_id='camera_init', color_field='intensity', cell_size=None):
    if point_count <= 0:
        raise ValueError('Cannot export an empty point cloud')
    if color_field not in ('intensity', 'rgb'):
        raise ValueError('Expected intensity or rgb as the fourth PCD field')
    frame = str(frame_id).replace('\n', ' ').replace('\r', ' ')
    cell_comment = ''
    if cell_size is not None:
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError('Cell size must be finite and positive')
        cell_comment = f'# cell_size: {cell_size}\n'
    return (f'# .PCD v0.7 - Point Cloud Data file format\n# frame_id: {frame}\n'
            f'{cell_comment}'
            f'VERSION 0.7\nFIELDS x y z {color_field}\nSIZE 4 4 4 4\n'
            'TYPE F F F F\nCOUNT 1 1 1 1\n'
            f'WIDTH {point_count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
            f'POINTS {point_count}\nDATA binary\n').encode('ascii', errors='replace')


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            write(output)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_pcd(path, points, frame_id='camera_init', color_field='intensity', cell_size=None):
    points = np.asarray(points, dtype='<f4')
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points[:, :3]).all():
        raise ValueError('Expected N x 4 points with finite XYZ coordinates')
    if color_field == 'intensity' and not np.isfinite(points[:, 3]).all():
        raise ValueError('Expected finite intensity values')
    header = pcd_header(len(points), frame_id, color_field, cell_size)

    def write(output):
        output.write(header)
        output.write(points.tobytes())
    _atomic_write(path, write)


def write_pcd_from_stream(path, raw_path, point_count, frame_id='camera_init', color_field='intensity'):
    """Wrap an accumulated XYZI stream without loading the whole map into RAM."""
    if Path(raw_path).stat().st_size != point_count * 16:
        raise ValueError('Point count does not match the accumulated point stream')
    header = pcd_header(point_count, frame_id, color_field)

    def write(output):
        output.write(header)
        with open(raw_path, 'rb') as source:
            shutil.copyfileobj(source, output)
    _atomic_write(path, write)


def generate_editor(pcd_path, html_path, frame_id='camera_init', cell_size=None, source_topic=''):
    """Embed a binary XYZI/XYZRGB PCD into an HTML file that works offline."""
    pcd_path = Path(pcd_path)
    with pcd_path.open('rb') as source:
        header = {}
        for _ in range(100):
            line = source.readline(4096).decode('ascii').strip()
            if line.startswith('# cell_size:') and cell_size is None:
                cell_size = float(line.split(':', 1)[1])
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition(' ')
            header[key] = value
            if key == 'DATA':
                break
        expected = {'SIZE': '4 4 4 4',
                    'TYPE': 'F F F F', 'COUNT': '1 1 1 1', 'DATA': 'binary'}
        if (header.get('FIELDS') not in ('x y z intensity', 'x y z rgb')
                or any(header.get(key) != value for key, value in expected.items())):
            raise ValueError('Editor requires binary float32 XYZ/intensity or XYZ/RGB PCD')
        count = int(header['POINTS'])
        if count <= 0 or pcd_path.stat().st_size - source.tell() != count * 16:
            raise ValueError('PCD point count does not match its data')
        template = Path(__file__).with_name('map_editor.html').read_text()
        meta = json.dumps(dict(name=pcd_path.name, frame_id=frame_id, point_count=count,
                               color_field=header['FIELDS'].split()[-1],
                               cell_size=cell_size, source_topic=source_topic))
        # Keep file names/frame names from closing an embedded script element.
        meta = meta.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        template = template.replace('__MAP_META_JSON__', meta)
        before, after = template.split('__POINT_DATA_BASE64__')

        def write(output):
            output.write(before.encode('utf-8'))
            # A multiple of three keeps concatenated base64 chunks valid.
            while True:
                chunk = source.read(3 * 1024 * 1024)
                if not chunk:
                    break
                output.write(base64.b64encode(chunk))
            output.write(after.encode('utf-8'))
        _atomic_write(html_path, write)
