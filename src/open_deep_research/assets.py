"""Register user-selected local assets; model tools only see opaque asset IDs."""
import hashlib
from pathlib import Path

import fitz


def register_image(run, path, asset_id='input_image', source_url=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 6_000_000:
        raise ValueError('Image must be a file smaller than 6 MB')
    raw = path.read_bytes()
    mime = 'image/png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if raw.startswith(b'\xff\xd8') else None
    if not mime:
        raise ValueError('Only PNG and JPEG images are supported')
    with fitz.open(stream=raw, filetype='png' if mime=='image/png' else 'jpeg') as doc:
        if doc[0].rect.width * doc[0].rect.height > 16_000_000:
            raise ValueError('Image dimensions exceed limit')
    digest = hashlib.sha256(raw).hexdigest()
    folder = run.directory/'assets'
    folder.mkdir(exist_ok=True)
    target = folder/(digest+('.png' if mime=='image/png' else '.jpg'))
    target.write_bytes(raw)
    record = {'id':asset_id, 'path':str(target), 'sha256':digest, 'mime':mime,
              'url':source_url or 'local-image:'+digest}
    run.images[asset_id] = record
    run.log('asset_registered', id=asset_id, sha256=digest, mime=mime)
    return record
