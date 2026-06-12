# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
datas += collect_data_files("fisheye", includes=["beam_widths/*.csv"])

hiddenimports = []
hiddenimports += collect_submodules("fisheye")
hiddenimports += collect_submodules("yolov5")

try:
    hiddenimports += collect_submodules("ultralytics")
except Exception:
    pass

runtime_hooks = ["fisheye_app/pyinstaller_cpu_runtime.py"]

a = Analysis(
    ["fisheye_app/__main__.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FisheyeArisSalmonDetection",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FisheyeArisSalmonDetection",
)
