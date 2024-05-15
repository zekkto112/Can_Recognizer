# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['install.py'],
    pathex=[],
    binaries=[],
    datas=[('pepsi_best.pt', '.'), ('coca_best.pt', '.'), ('7up_best.pt', '.'), ('yolov8n.pt', '.'), ('default.yaml', 'ultralytics/cfg')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='install',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
